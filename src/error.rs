use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::Json;
use thiserror::Error;

use crate::types::{ModelId, NodeId, ShardId};

#[derive(Error, Debug)]
pub enum SwarmError {
    // Network
    #[error("Network error: {0}")]
    Network(String),
    #[error("Peer not found: {0}")]
    PeerNotFound(NodeId),

    // Inference
    #[error("Model not available: {0}")]
    ModelNotAvailable(ModelId),
    #[error("Inference error: {0}")]
    Inference(String),
    #[error("Insufficient network capacity for model {0}")]
    InsufficientCapacity(ModelId),
    #[error("Pipeline assembly failed: {0}")]
    PipelineError(String),
    /// A peer took this request and then went silent — no acknowledgement, no
    /// first token, no explicit failure — until the delivery watchdog gave up
    /// (`RR_ACK_TIMEOUT_SECS` sweep, or the prompt-scaled first-token
    /// deadline).
    ///
    /// Transient and NOT this node's bug, which is why it is its own variant:
    /// as a `PipelineError` it answered `500 server_error` — a vanished peer
    /// reported as a bug in the node the user was talking to — and inherited
    /// `PipelineError`'s exemption from `failure_is_penalty_worthy`, so the
    /// peer that went silent was never docked even though "timeouts waiting on
    /// a peer" is exactly what that penalty exists for. The router retries a
    /// fresh assembly once before this surfaces, so a caller seeing it has had
    /// two attempts go quiet; a new request usually routes to a different
    /// holder, which is what the hint says.
    #[error("Peer unresponsive: {0}")]
    PeerUnresponsive(String),
    /// A peer serving one segment of a distributed pipeline failed
    /// mid-request and no hot-standby covered its layer range, so the request
    /// could not continue.
    ///
    /// 503, not 500: nothing is wrong with this node or with the caller's
    /// request — there was nobody free to take the segment over (observed
    /// live 2026-08-15 with two holders of a range, one busy; the manual
    /// retry succeeded once the peer was idle). Deliberately NOT
    /// `ModelIncompleteInSwarm`, whose variant makes
    /// `assembly_failed_for_lack_of_holders` wait on DHT results — pointless
    /// here, the holders are known and one just failed — and NOT
    /// `ServiceUnavailable`, whose wording marks a PEER as unable to serve
    /// and triggers the blacklist-retry. Credit attribution stays with the
    /// underlying segment failure, never with this summary (it names no
    /// culprit).
    #[error("Segment failover exhausted: {0}")]
    SegmentFailoverExhausted(String),
    /// A peer produced the whole answer and part of it was lost on the way
    /// here, so what arrived is not the reply that was generated.
    ///
    /// Every reply token is an independent fire-and-forget send with no
    /// acknowledgement and no retransmission, so one drop truncates the answer
    /// permanently — the reassembler may only release the consecutive run,
    /// because emitting past a hole would silently reorder the reply. Measured
    /// 2026-08-20: replies came back holding 3, 3 and 18 of 60 tokens.
    ///
    /// **This exists because the alternative was a lie.** Such a reply used to
    /// be handed to the caller as a normal completion with
    /// `finish_reason: "stop"` — i.e. the model chose to stop after three
    /// tokens. A client cannot tell that from a real answer.
    ///
    /// 503, not 500: nothing is wrong with this node, with the caller's request,
    /// or with the peer's hardware. Deliberately NOT `PeerUnresponsive`, whose
    /// whole meaning is that a peer went quiet — here it answered in full and
    /// the network dropped part of it, and blaming the peer for that is the
    /// wrong-culprit mistake this project has made twice. Penalty-exempt for the
    /// same reason, and deliberately NOT in `is_transient_remote_failure`: the
    /// retry there reuses the caller's token channel, so retrying a streaming
    /// request would emit its reply twice.
    #[error("Reply truncated in transit: {0}")]
    ReplyTruncated(String),
    #[error("Inference timeout after {0}s")]
    InferenceTimeout(u64),
    #[error("No model loaded")]
    NoModelLoaded,
    #[error("No vision encoder (mmproj) available for model {0}")]
    VisionEncoderUnavailable(ModelId),

    // Shards
    #[error("Shard verification failed: expected {expected}, got {actual}")]
    ShardIntegrity { expected: String, actual: String },
    /// The shard file is the wrong SIZE — the transfer did not complete.
    ///
    /// Distinct from `ShardIntegrity`, which means the bytes arrived in full
    /// but are not the bytes we asked for. Only the latter is evidence about
    /// the sender; conflating them penalises honest peers on a flaky link.
    #[error("Shard transfer incomplete: expected {expected_bytes} bytes, got {actual_bytes}")]
    ShardIncomplete {
        expected_bytes: u64,
        actual_bytes: u64,
    },
    #[error("Shard not found: {0:?}")]
    ShardNotFound(ShardId),

    // Credits
    #[error("Insufficient credits: balance={balance}, required={required}")]
    InsufficientCredits { balance: i64, required: i64 },
    #[error("Invalid transaction signature")]
    InvalidSignature,
    #[error("Credit error: {0}")]
    CreditError(String),

    // Encryption
    #[error("Encryption error: {0}")]
    Encryption(String),
    #[error("Decryption failed (invalid key, corrupted data, or tampered ciphertext)")]
    DecryptionFailed,
    #[error("No encryption session for peer: {0}")]
    NoSession(NodeId),
    #[error("Nonce counter overflow — session must be re-established")]
    NonceOverflow,

    // Identity
    #[error("Keystore error: {0}")]
    Keystore(String),
    #[error("Invalid nickname: {0}")]
    InvalidNickname(String),

    // Storage
    #[error("Database error: {0}")]
    Database(String),
    #[error("Insufficient disk space: need {need_mb}MB, have {have_mb}MB")]
    InsufficientDisk { need_mb: u64, have_mb: u64 },

    // Auth
    #[error("Unauthorized: {0}")]
    Unauthorized(String),

    /// Refused because the action must be performed ON the machine running this
    /// node — NOT because the caller failed to authenticate.
    ///
    /// Kept separate from `Unauthorized` because the two need opposite advice
    /// and only one of them is fixable by the caller. `Unauthorized`'s hint
    /// sends the user to the dashboard to fetch their API key, which is exactly
    /// right for a missing key and useless here: the caller already sent a valid
    /// one, so following that hint loops forever. This is gotcha #295's shape —
    /// a permanent failure handed advice that cannot resolve it — and the reason
    /// a permanent refusal gets its own variant rather than a string stuffed
    /// into a general one.
    ///
    /// Holds the action, capitalised, e.g. `"Applying an update"`.
    #[error("{0} can only be done on the computer running this node")]
    LocalOnly(String),

    // Validation
    #[error("Validation error: {0}")]
    Validation(String),

    // Resource lookup miss (generic 404 — distinct from Validation 400). Use
    // for endpoints where the request shape is fine but the named resource
    // (response id, session id, etc.) doesn't exist in the store.
    #[error("Not found: {0}")]
    NotFound(String),

    // Config
    #[error("Configuration error: {0}")]
    Config(String),

    // Provider proxy
    #[error("Provider error ({status}): {body}")]
    ProviderError { status: u16, body: String },

    // Private mode
    #[error("Private mode: model {model_id} not fully available in your device pool (missing shards: {missing_shards:?})")]
    PrivateModeUnavailable {
        model_id: String,
        missing_shards: Vec<u32>,
    },

    /// Prompt privacy (`inference.encrypted_pipeline`) is on for this model, but
    /// this node does not hold shard 0 — the embedding table that has to stay
    /// local for the prompt to be hidden from every peer.
    ///
    /// This is a POLICY refusal, not a fault: the setting and the shards on disk
    /// disagree, commonly because privacy was switched on while shard 0 was
    /// present and the shard was later pruned. Retrying can never resolve it,
    /// which is why it is its own variant rather than a `PipelineError` — as the
    /// latter it inherited a generic "a peer went offline, try again" hint and
    /// sent users round a loop that had no exit.
    #[error("Prompt privacy is on for {model_id}, but this node does not hold shard 0 (the embedding table) that keeps your prompt away from every peer")]
    PromptPrivacyUnavailable { model_id: String },

    /// No reachable node holds the piece of this model covering `layer` — the
    /// swarm is missing part of it, so no pipeline can be assembled.
    ///
    /// A capacity problem, not a fault in this server, and the sibling of
    /// `InsufficientCapacity`: that one fires when NOTHING is available and
    /// already answered 503, while this one fired when SOME shards were missing
    /// and answered 500. Two readings of one situation — "the swarm hasn't got
    /// all of this model" — differing only by how much was absent.
    #[error("No reachable node holds the part of {model_id} containing layer {layer}. A model can be listed, and even loaded here, while the peer that held that piece has gone")]
    ModelIncompleteInSwarm { model_id: String, layer: u32 },

    // Overload
    #[error("Service unavailable: {0}")]
    ServiceUnavailable(String),

    /// This build does not implement the thing that was asked for.
    ///
    /// Distinct from `ServiceUnavailable`, which means "not right now" and
    /// invites a retry. This one will never succeed however long you wait, so
    /// it must not wear a 503: a client with retry-on-5xx re-sends forever, and
    /// monitoring reads a permanent capability gap as this node being unwell.
    /// 501 says the honest thing; the message must name what to do instead.
    #[error("Not implemented: {0}")]
    NotImplemented(String),

    // Generic
    #[error("Internal error: {0}")]
    Internal(String),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Serialization(#[from] serde_json::Error),
}

impl SwarmError {
    /// Build `SwarmError::Internal` from any `Display`-able error. Use as
    /// `.map_err(SwarmError::internal)` to avoid the closure boilerplate.
    pub fn internal<E: std::fmt::Display>(e: E) -> Self {
        SwarmError::Internal(e.to_string())
    }
}

/// API-facing error that maps SwarmError to HTTP status codes.
pub struct ApiError(pub SwarmError);

impl From<SwarmError> for ApiError {
    fn from(err: SwarmError) -> Self {
        ApiError(err)
    }
}

/// Recover the CLASS of an error from a message that crossed a boundary which
/// does not carry types.
///
/// `SwarmError` survives neither the worker IPC hop nor the network hop — both
/// deliver a `String`. Whatever is left is re-wrapped as `Inference`, i.e. HTTP
/// 500 `server_error`, so a failure the caller could act on, or that another
/// peer could have served, is reported as this server having broken.
///
/// Both boundaries had the same problem; only the worker one had the remedy.
/// Measured 2026-08-12: an over-long prompt sent to a model held only by peers
/// came back `500 server_error: Inference error: Validation error: This
/// conversation is too long …` — the peer had diagnosed it exactly right, and
/// the diagnosis arrived wearing the wrong status. Locally the identical
/// request is a `400 invalid_request_error`.
///
/// `rfind` takes the INNERMOST marker so a message wrapped more than once
/// (worker → pipeline → router → peer) still yields the original reason rather
/// than a fragment of an outer wrapper. Precedence is fixed and matches the
/// order these are checked at the worker boundary.
///
/// This is deliberately matching on prose, which is normally the trap in gotcha
/// #295 — it is sound *only* because the markers are `SwarmError`'s own
/// `#[error(...)]` Display prefixes, which are part of the type, not
/// user-facing wording that gets rewritten. Adding a variant here means adding
/// its marker, and nothing else re-derives a class from a message.
pub fn reclassify_flattened_error(message: &str) -> Option<SwarmError> {
    fn detail_after(message: &str, marker: &str) -> Option<String> {
        let idx = message.rfind(marker)?;
        let detail = message[idx + marker.len()..].trim();
        (!detail.is_empty()).then(|| detail.to_string())
    }

    if let Some(d) = detail_after(message, "Validation error: ") {
        return Some(SwarmError::Validation(d));
    }
    if let Some(d) = detail_after(message, "Model not available: ") {
        return Some(SwarmError::ModelNotAvailable(ModelId(d)));
    }
    if let Some(d) = detail_after(message, "Service unavailable: ") {
        return Some(SwarmError::ServiceUnavailable(d));
    }
    if let Some(d) = detail_after(message, "Peer unresponsive: ") {
        return Some(SwarmError::PeerUnresponsive(d));
    }
    if let Some(d) = detail_after(message, "Segment failover exhausted: ") {
        return Some(SwarmError::SegmentFailoverExhausted(d));
    }
    None
}

/// Classify an error into (HTTP status, client-safe message, error type).
///
/// The single definition of what an error *is* to a caller. Extracted from
/// `ApiError::into_response` so the STREAMING paths can label an SSE error
/// frame with the same type the non-streaming sibling returns for the identical
/// failure.
///
/// They could not, so they hardcoded one: an over-long prompt came back as
/// `invalid_request_error` with a 400 when the client didn't stream, and as
/// `server_error` inside a 200 when it did — the same user mistake reported as
/// this server breaking (measured 2026-08-12). Anything that needs to name an
/// error to a client goes through here rather than choosing a type locally.
pub fn classify_error(err: &SwarmError) -> (StatusCode, String, &'static str) {
    match err {
        SwarmError::ModelNotAvailable(_) => {
            (StatusCode::NOT_FOUND, err.to_string(), "not_found_error")
        }
        SwarmError::NoModelLoaded => (
            StatusCode::SERVICE_UNAVAILABLE,
            err.to_string(),
            "server_error",
        ),
        SwarmError::InferenceTimeout(_) => {
            (StatusCode::GATEWAY_TIMEOUT, err.to_string(), "server_error")
        }
        SwarmError::InsufficientCredits { .. } => (
            StatusCode::PAYMENT_REQUIRED,
            err.to_string(),
            "insufficient_credits",
        ),
        SwarmError::NotImplemented(_) => (
            StatusCode::NOT_IMPLEMENTED,
            err.to_string(),
            "not_implemented_error",
        ),
        SwarmError::InsufficientCapacity(_) | SwarmError::ServiceUnavailable(_) => (
            StatusCode::SERVICE_UNAVAILABLE,
            err.to_string(),
            "server_error",
        ),
        SwarmError::PrivateModeUnavailable { .. } => (
            StatusCode::SERVICE_UNAVAILABLE,
            err.to_string(),
            "private_mode_error",
        ),
        SwarmError::PromptPrivacyUnavailable { .. } => (
            StatusCode::SERVICE_UNAVAILABLE,
            err.to_string(),
            "prompt_privacy_error",
        ),
        // The swarm is missing part of this model: "this server can't
        // serve", exactly like its sibling `InsufficientCapacity`. It
        // answered 500 until 2026-08-11, reporting a capacity shortfall as
        // a fault in the node the user is talking to.
        SwarmError::ModelIncompleteInSwarm { .. } => (
            StatusCode::SERVICE_UNAVAILABLE,
            err.to_string(),
            "server_error",
        ),
        SwarmError::VisionEncoderUnavailable(_) => (
            StatusCode::SERVICE_UNAVAILABLE,
            err.to_string(),
            "server_error",
        ),
        SwarmError::InsufficientDisk { .. } => (
            StatusCode::INSUFFICIENT_STORAGE,
            err.to_string(),
            "server_error",
        ),
        SwarmError::ShardIntegrity { .. } => (
            StatusCode::INTERNAL_SERVER_ERROR,
            err.to_string(),
            "server_error",
        ),
        SwarmError::ShardIncomplete { .. } => (
            StatusCode::SERVICE_UNAVAILABLE,
            err.to_string(),
            "service_unavailable",
        ),
        SwarmError::PipelineError(_) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            err.to_string(),
            "server_error",
        ),
        // A peer going quiet is "this server couldn't serve it just now", not
        // a bug in this server — the 503 invites the retry that actually
        // helps (a fresh request usually routes to a different holder).
        SwarmError::PeerUnresponsive(_) => (
            StatusCode::SERVICE_UNAVAILABLE,
            err.to_string(),
            "server_error",
        ),
        // Same reasoning for a mid-pipeline holder failure with nobody free
        // to take the segment over.
        SwarmError::SegmentFailoverExhausted(_) => (
            StatusCode::SERVICE_UNAVAILABLE,
            err.to_string(),
            "server_error",
        ),
        // The peer answered in full; the network lost part of it. 503 invites
        // the retry that helps, and a fresh request usually routes elsewhere.
        SwarmError::ReplyTruncated(_) => (
            StatusCode::SERVICE_UNAVAILABLE,
            err.to_string(),
            "server_error",
        ),
        SwarmError::Inference(_) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            err.to_string(),
            "server_error",
        ),
        SwarmError::ProviderError { status, ref body } => {
            let http_status = StatusCode::from_u16(*status).unwrap_or(StatusCode::BAD_GATEWAY);
            // Truncate provider body to avoid leaking upstream internals
            let safe_body: String = body.chars().take(512).collect();
            (
                http_status,
                format!("Provider error: {safe_body}"),
                "server_error",
            )
        }
        SwarmError::PeerNotFound(_) => (
            StatusCode::SERVICE_UNAVAILABLE,
            err.to_string(),
            "server_error",
        ),
        SwarmError::Unauthorized(_) => (
            StatusCode::UNAUTHORIZED,
            err.to_string(),
            "authentication_error",
        ),
        // 403, not 401: the caller IS authenticated. 401 tells them their
        // credentials were not accepted, which sends them off to re-check a key
        // that was fine all along.
        SwarmError::LocalOnly(_) => (StatusCode::FORBIDDEN, err.to_string(), "permission_error"),
        // SwarmError::Config is for daemon startup / config-file errors per
        // .claude/rules/completeness.md. If it surfaces in an HTTP response
        // path, the daemon has shipped misconfigured — that's a 500, not a
        // 400 (the user did not send invalid input).
        SwarmError::Config(_) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            err.to_string(),
            "server_error",
        ),
        SwarmError::InvalidNickname(_) | SwarmError::Validation(_) => (
            StatusCode::BAD_REQUEST,
            err.to_string(),
            "invalid_request_error",
        ),
        SwarmError::ShardNotFound(_) => (StatusCode::NOT_FOUND, err.to_string(), "not_found_error"),
        SwarmError::NotFound(_) => (StatusCode::NOT_FOUND, err.to_string(), "not_found_error"),
        // Network upstream-unreachable: 502 + a distinct error_type so
        // SDK retry logic can distinguish 'try later' from a 500 bug.
        SwarmError::Network(_) => (StatusCode::BAD_GATEWAY, err.to_string(), "network_error"),
        _ => {
            // Return a generic message to avoid leaking internal paths, peer
            // errors, or DB details. The FULL error is still recorded — by
            // whoever logs this failure, from the original `SwarmError` rather
            // than from this genericised string.
            //
            // This arm used to `tracing::error!` here. That made the
            // classifier impure, and a classifier with a logging side effect
            // cannot be consulted by a logging path without emitting a line of
            // its own — which is exactly what `failure_log_level` below needs
            // to do.
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                "An internal error occurred".to_string(),
                "server_error",
            )
        }
    }
}

/// How loudly a failure deserves to be recorded in **this node's** log.
///
/// The answer is already implied by the status `classify_error` picks, because
/// that status *is* the answer to "whose mistake was this". Deriving the level
/// from it, rather than choosing one per call site, is what keeps an operator's
/// log honest.
///
/// **Why this exists.** A caller's own too-long prompt was recorded as three
/// separate `ERROR` lines when the model happened to be peer-held, and one
/// `WARN` when it was local — the same user mistake, at a different severity,
/// depending on which machine held the model. A `501` for embeddings (a
/// deliberate, documented property of this build, answered with a helpful
/// message) was logged as `ERROR Server error`. `ERROR` in a log means "this
/// node is broken"; for a non-technical operator deciding whether to trust this
/// software, reporting their own typo that way is worse than saying nothing.
/// It is the logging-layer survivor of the class fixed at the HTTP surface in
/// gotchas #300-#305.
///
/// Driving it off the status keeps it in lockstep by construction: a new
/// `SwarmError` variant inherits a sensible level from the status it already
/// had to choose, with no second decision to forget.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FailureLevel {
    /// Ours. A bug, or something genuinely broken on this node.
    Error,
    /// Real, but not a bug: a peer went quiet, an upstream provider failed,
    /// capacity ran out. Worth an operator's attention, not an alarm.
    Warn,
    /// Working as designed — the caller's own mistake, or a deliberate policy
    /// or build property. The response already told them; the log does not
    /// need to shout about it.
    Info,
}

/// Pick the log level for a failure. See [`FailureLevel`].
pub fn failure_log_level(err: &SwarmError) -> FailureLevel {
    let (status, _, _) = classify_error(err);

    // The caller's mistake. Their response said so; this node is fine.
    if status.is_client_error() {
        return FailureLevel::Info;
    }

    match status {
        // A deliberate property of this build, answered with a message that
        // says what to do instead. Not a fault.
        StatusCode::NOT_IMPLEMENTED => FailureLevel::Info,
        // Genuinely ours: `Internal`, `Config` reaching a request path,
        // `ShardIntegrity`, and the catch-all.
        StatusCode::INTERNAL_SERVER_ERROR => FailureLevel::Error,
        // Everything else in 5xx names an external party or a transient
        // condition: a silent peer, an exhausted failover, a provider outage,
        // a full disk. Real, but not this node malfunctioning.
        _ => FailureLevel::Warn,
    }
}

/// Record a failed request at the severity its cause deserves.
///
/// Takes the `SwarmError` first, then the ordinary `tracing` field/message
/// syntax. Use this instead of picking `error!` / `warn!` at the call site —
/// the level is not a call-site decision, for the reasons on [`FailureLevel`].
///
/// ```ignore
/// log_failure!(e, request_id = %id, model = %m, "inference failed");
/// ```
#[macro_export]
macro_rules! log_failure {
    ($err:expr, $($rest:tt)*) => {{
        $crate::log_at_level!($crate::error::failure_log_level($err), $($rest)*)
    }};
}

/// Emit a `tracing` event at an already-decided [`FailureLevel`].
///
/// [`log_failure!`] is the form to reach for, because it derives the level from
/// a `SwarmError` and so cannot be got wrong. This one exists for the surfaces
/// whose errors are **strings all the way down** and have their own predicate
/// for whose fault a failure was — the HuggingFace client is the case: it
/// returns `Result<_, String>`, and `probe_failure_is_user_fixable` is where
/// that surface decides. The point is still that ONE place decides and the call
/// site only reports.
#[macro_export]
macro_rules! log_at_level {
    ($level:expr, $($rest:tt)*) => {{
        match $level {
            $crate::error::FailureLevel::Error => tracing::error!($($rest)*),
            $crate::error::FailureLevel::Warn => tracing::warn!($($rest)*),
            $crate::error::FailureLevel::Info => tracing::info!($($rest)*),
        }
    }};
}

impl IntoResponse for ApiError {
    fn into_response(self) -> axum::response::Response {
        let (status, message, error_type) = classify_error(&self.0);

        // Record every failure the API returns, at the severity its cause
        // deserves — see `FailureLevel`. Logs `self.0`, not `message`: the
        // catch-all genericises the message to keep internals out of the
        // response, so logging the message would record "An internal error
        // occurred" and throw away the only copy of what actually happened.
        crate::log_failure!(
            &self.0,
            status = status.as_u16(),
            error = %self.0,
            "API request failed"
        );

        let mut error_obj = serde_json::json!({
            "message": message,
            "type": error_type,
            "param": null,
            "code": error_type
        });
        // `hint` stays exactly as it was — English prose, for API clients and
        // for anything that cannot translate. `hint_key` is additive: it is
        // what the dashboard looks up to show the same advice in the user's own
        // language, and what a client can branch on without matching prose.
        if let Some((key, hint_text)) = error_hint_with_key(&self.0) {
            error_obj["hint"] = serde_json::Value::String(hint_text.to_string());
            error_obj["hint_key"] = serde_json::Value::String(key.to_string());
        }

        (status, Json(serde_json::json!({ "error": error_obj }))).into_response()
    }
}

/// Return an actionable hint for common error variants, as a stable
/// `(key, english)` pair.
///
/// These are user-facing — no curl commands or API paths.
///
/// **Why both, from one function.** These hints were the only user-facing text
/// in the product with no translation route at all: the dashboard ships 21
/// locales and every one of them showed these paragraphs in English. The key is
/// what lets the frontend translate them (`I18n.t("error_hint." + key)`, with
/// the English carried alongside as the fallback), and what an API client can
/// branch on instead of matching prose.
///
/// Returning the pair from a single match arm is deliberate. A separate
/// `error_hint_key` function would be a second decision to keep in step, and
/// this codebase's most-repeated defect is exactly that — one invariant
/// implemented per path. Here they cannot drift: only one place knows either.
pub fn error_hint_with_key(err: &SwarmError) -> Option<(&'static str, &'static str)> {
    match err {
        SwarmError::ModelNotAvailable(_) => Some((
            "model_not_available",
            "This model isn't available yet. Open the Models tab in the dashboard to browse \
             and download models, or wait for auto-manage to acquire it from the network.",
        )),
        SwarmError::NoModelLoaded => Some((
            "no_model_loaded",
            "No model is loaded yet. Go to the dashboard and select a model to download, \
             or connect to more peers so models can be served from the network.",
        )),
        SwarmError::InsufficientCredits { .. } => Some((
            "insufficient_credits",
            "Your credit balance is too low. Earn credits by hosting model shards \
             (happens automatically) or serving inference for other users. \
             Check your balance on the dashboard.",
        )),
        SwarmError::InsufficientCapacity(_) => Some((
            "insufficient_capacity",
            "Not enough peers have the shards needed for this model. \
             Try again later as more peers come online, or download the model \
             shards yourself from the Models tab.",
        )),
        SwarmError::InsufficientDisk { .. } => Some((
            "insufficient_disk",
            "Not enough disk space. Free up space or increase the storage limit \
             in Settings → Advanced → max_disk_mb.",
        )),
        SwarmError::Unauthorized(_) => Some((
            "unauthorized",
            "Authentication required. Your API key can be found on the dashboard \
             Settings page. Include it as a Bearer token in the Authorization header.",
        )),
        // Deliberately says nothing about API keys: the caller already has a
        // working one, and repeating the Unauthorized advice here is what sent
        // remote admins to re-copy a key that was never the problem.
        // Says why WITHOUT naming a mechanism: this covers replacing the
        // binary, downloading it, and shutting the node down, and "it writes to
        // that machine's disk" was false for the last one.
        SwarmError::LocalOnly(_) => Some((
            "local_only",
            "This one has to be done on the computer that is running SwarmLLM, \
             because it acts on that machine itself. Open the dashboard there \
             and try again — your API key is fine.",
        )),
        SwarmError::PeerNotFound(_) => Some((
            "peer_not_found",
            "That peer is offline or unreachable. Check your internet connection \
             and try again later.",
        )),
        SwarmError::ShardIntegrity { .. } => Some((
            "shard_integrity",
            "A model file was corrupted and will be re-downloaded automatically. \
             This is usually caused by an interrupted download — try again in a moment.",
        )),
        SwarmError::ShardIncomplete { .. } => Some((
            "shard_incomplete",
            "A model file did not download completely and will be fetched again \
             automatically. This usually means the connection dropped part-way.",
        )),
        // `PipelineError` covers two causes needing OPPOSITE advice, and the
        // generic text asserted the wrong one. A tester was told "a peer went
        // offline mid-request" when the real cause was a piece of the model
        // missing locally that no peer could supply — so following the hint
        // means retrying for ever instead of fetching the piece. Reported
        // 2026-07-30.
        SwarmError::PipelineError(msg) if msg.contains("No node available for layer") => Some((
            "pipeline_missing_layer",
            "Part of this model isn't available — not on this machine, and not on any \
             connected peer. Fetch the whole model with `swarmllm get-model <name> --all`, \
             or wait for more peers to come online.",
        )),
        // The remaining pipeline failures are a mix of transient and permanent
        // causes, and we do not know which this one is. The old wording picked
        // the transient reading and stated it as fact — "a peer went offline …
        // try again" — so every permanent cause sent the user round a loop with
        // no exit. Say what is known, offer both branches, promise neither.
        SwarmError::PipelineError(_) => Some((
            "pipeline_generic",
            "The route to run this model couldn't be put together. If a peer dropped out \
             this will fix itself — try once more. If it fails the same way again, the \
             model is missing a piece: fetch it with `swarmllm get-model <name> --all`, \
             or pick a model marked as ready in the dashboard.",
        )),
        // Unlike `PipelineError`, this one is KNOWN to be transient — a peer
        // took the work and went silent — so the hint can promise the retry
        // branch without hedging.
        // Deliberately does NOT assert that the peer went offline. A tester
        // reported this exact hint for a machine that was up and had answered
        // immediately, twice, with a precise reason — its reply was lost on the
        // way, and the terminal frame carrying it is a single unacknowledged
        // send just like every content token. Naming a cause we cannot observe
        // sends people to check the wrong thing.
        SwarmError::PeerUnresponsive(_) => Some((
            "peer_unresponsive",
            "Another computer in the swarm took this request and no reply \
             arrived in time. It may have gone offline, or its answer may have \
             been lost on the way. Try again: a new request is usually routed \
             to a different machine.",
        )),
        SwarmError::ReplyTruncated(_) => Some((
            "reply_truncated",
            "Part of the answer was lost travelling back from the computer that \
             produced it, so what arrived was incomplete and has not been shown \
             as if it were the whole reply. Try again: a new request is usually \
             routed to a different machine.",
        )),
        SwarmError::SegmentFailoverExhausted(_) => Some((
            "segment_failover_exhausted",
            "A computer running part of this model failed mid-request, and no \
             other machine was free to take over its part. This is usually \
             momentary — try again, and the swarm will build a fresh route.",
        )),
        SwarmError::ModelIncompleteInSwarm { .. } => Some((
            "model_incomplete_in_swarm",
            "Part of this model isn't on any machine that's reachable right now. If a peer \
             just dropped out this may fix itself shortly. Otherwise fetch the model with \
             `swarmllm get-model <name> --all`, or pick a model the dashboard marks as ready.",
        )),
        SwarmError::PromptPrivacyUnavailable { .. } => Some((
            "prompt_privacy_unavailable",
            "Prompt privacy keeps your prompt on this machine, which needs the model's \
             first part stored here — and it isn't. Either fetch it with \
             `swarmllm get-model <name>`, or turn prompt privacy off for this model to \
             let the swarm run it. Retrying as-is won't help.",
        )),
        // The only refusal that had no hint, and its message is the least
        // readable of them: "missing shards: [0, 1, 2, 3]" tells a non-technical
        // user nothing, and nothing else on screen says what to do about it.
        //
        // Turning private mode off is deliberately listed LAST and with its
        // consequence stated. It is the quickest way to make the error go away
        // and the only one that changes where the user's prompts travel, so
        // offering it casually would trade someone's privacy for convenience
        // without telling them.
        SwarmError::PrivateModeUnavailable { .. } => Some((
            "private_mode_unavailable",
            "Private mode keeps your prompts on your own devices, and none of them has \
             all of this model. Either download it here with `swarmllm get-model <name>`, \
             or add the device that has it to your pool. You can also turn private mode \
             off in Settings — but then your prompts can be sent to other people's \
             machines to run.",
        )),
        SwarmError::InferenceTimeout(_) => Some((
            "inference_timeout",
            "The request took too long. Try a shorter prompt, reduce the max tokens, \
             or wait for a less busy time.",
        )),
        SwarmError::Config(_) => Some((
            "config",
            "There's a configuration issue. Check Settings in the dashboard \
             or review your config.toml file.",
        )),
        SwarmError::ProviderError { status, ref body } => {
            let lower = body.to_lowercase();
            let is_quota = *status == 402
                || *status == 429
                || lower.contains("quota")
                || lower.contains("billing")
                || lower.contains("insufficient")
                || lower.contains("balance")
                || lower.contains("exceeded")
                || lower.contains("limit")
                || lower.contains("credits")
                || lower.contains("payment");
            if is_quota {
                Some((
                    "provider_quota",
                    "Your cloud provider credits may be exhausted or rate-limited. \
                     Top up your account on the provider's website, switch to a free-tier provider \
                     (DeepSeek, Groq, NVIDIA NIM), or use a local swarm model instead.",
                ))
            } else if *status == 401 || *status == 403 {
                Some((
                    "provider_auth",
                    "Your API key appears to be invalid or revoked. \
                     Update it in Settings → Cloud Providers.",
                ))
            } else {
                Some((
                    "provider_generic",
                    "The cloud provider returned an error. Try again, \
                     switch to a different model, or use a local swarm model.",
                ))
            }
        }
        _ => None,
    }
}

/// The English hint text alone. Prefer [`error_hint_with_key`] where the key is
/// useful (the HTTP envelope carries it so the dashboard can translate).
pub fn error_hint(err: &SwarmError) -> Option<&'static str> {
    error_hint_with_key(err).map(|(_, text)| text)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hint_for_model_not_available() {
        let err = SwarmError::ModelNotAvailable(ModelId("test-model".into()));
        assert!(error_hint(&err).unwrap().contains("Models tab"));
    }

    /// Every hint key must be distinct. Two variants sharing one key would show
    /// the wrong translated advice for one of them while the English stayed
    /// right — a divergence visible only to users not reading English, which is
    /// the group this whole mechanism exists for.
    #[test]
    fn hint_keys_are_unique() {
        let samples: Vec<SwarmError> = vec![
            SwarmError::ModelNotAvailable(ModelId("m".into())),
            SwarmError::NoModelLoaded,
            SwarmError::InsufficientCredits {
                balance: 0,
                required: 1,
            },
            SwarmError::InsufficientCapacity(ModelId("m".into())),
            SwarmError::InsufficientDisk {
                need_mb: 1,
                have_mb: 0,
            },
            SwarmError::Unauthorized("x".into()),
            SwarmError::LocalOnly("x".into()),
            SwarmError::PeerNotFound(NodeId([0u8; 32])),
            SwarmError::ShardIntegrity {
                expected: "a".into(),
                actual: "b".into(),
            },
            SwarmError::ShardIncomplete {
                expected_bytes: 2,
                actual_bytes: 1,
            },
            SwarmError::PipelineError("No node available for layer 3".into()),
            SwarmError::PipelineError("something else".into()),
            SwarmError::PeerUnresponsive("x".into()),
            SwarmError::SegmentFailoverExhausted("x".into()),
            SwarmError::ModelIncompleteInSwarm {
                model_id: "m".to_string(),
                layer: 3,
            },
            SwarmError::PromptPrivacyUnavailable {
                model_id: "m".into(),
            },
            SwarmError::PrivateModeUnavailable {
                model_id: "m".into(),
                missing_shards: vec![0],
            },
            SwarmError::InferenceTimeout(1),
            SwarmError::Config("x".into()),
            SwarmError::ProviderError {
                status: 429,
                body: "quota".into(),
            },
            SwarmError::ProviderError {
                status: 401,
                body: "bad key".into(),
            },
            SwarmError::ProviderError {
                status: 500,
                body: "boom".into(),
            },
        ];

        let mut seen = std::collections::HashSet::new();
        for err in &samples {
            let (key, text) = error_hint_with_key(err)
                .unwrap_or_else(|| panic!("every sampled variant has a hint: {err}"));
            assert!(
                seen.insert(key),
                "duplicate hint key {key:?} — two variants would share one translation"
            );
            assert!(
                !key.is_empty() && key.chars().all(|c| c.is_ascii_lowercase() || c == '_'),
                "hint keys are lower_snake_case i18n identifiers, got {key:?}"
            );
            assert!(!text.is_empty(), "hint {key:?} has no English text");
        }
        assert_eq!(seen.len(), samples.len());
    }

    /// `error_hint` must stay a view of `error_hint_with_key`, never a second
    /// copy of the text.
    #[test]
    fn the_english_hint_comes_from_the_keyed_one() {
        let err = SwarmError::NoModelLoaded;
        assert_eq!(error_hint(&err), error_hint_with_key(&err).map(|(_, t)| t));
    }

    /// A caller's own mistake is not this node breaking, and the log must not
    /// say it is.
    ///
    /// The over-long prompt is the measured case (2026-08-17, live v0.3.99
    /// node): it produced three `ERROR` lines when the model happened to be
    /// peer-held and one `WARN` when it was local — the same user typo, at a
    /// different severity, decided by which machine held the model.
    #[test]
    fn a_callers_mistake_is_not_logged_as_this_node_failing() {
        for err in [
            SwarmError::Validation("conversation is too long".into()),
            SwarmError::InvalidNickname("bad".into()),
            SwarmError::ModelNotAvailable(ModelId("nope".into())),
            SwarmError::NotFound("nope".into()),
            SwarmError::Unauthorized("no key".into()),
            SwarmError::LocalOnly("loopback only".into()),
        ] {
            assert_eq!(
                failure_log_level(&err),
                FailureLevel::Info,
                "a 4xx is the caller's mistake, not an error on this node: {err}"
            );
        }
    }

    /// A deliberate property of this build, answered with a message saying what
    /// to do instead, logged as `ERROR Server error` on the live node.
    #[test]
    fn a_deliberate_refusal_is_not_an_error() {
        assert_eq!(
            failure_log_level(&SwarmError::NotImplemented("no embeddings".into())),
            FailureLevel::Info,
        );
        // Policy refusals are configuration doing its job, not a fault. They
        // are 503s, so they land on the Warn arm rather than Info — still not
        // ERROR, which is the property that matters here.
        for err in [
            SwarmError::PrivateModeUnavailable {
                model_id: "m".to_string(),
                missing_shards: vec![],
            },
            SwarmError::PromptPrivacyUnavailable {
                model_id: "m".to_string(),
            },
        ] {
            assert_ne!(failure_log_level(&err), FailureLevel::Error, "{err}");
        }
    }

    /// The converse: our own bugs must still be loud. A rule that only ever
    /// quietens things would "fix" this by hiding real faults.
    #[test]
    fn our_own_faults_are_still_errors() {
        for err in [
            SwarmError::Internal("bug".into()),
            SwarmError::Config("misconfigured".into()),
            SwarmError::Inference("worker blew up".into()),
        ] {
            assert_eq!(failure_log_level(&err), FailureLevel::Error, "{err}");
        }
    }

    /// A peer going quiet is real and worth seeing, but it is not this node
    /// malfunctioning — it must sit between the two.
    #[test]
    fn someone_elses_fault_is_a_warning_not_an_error() {
        for err in [
            SwarmError::PeerUnresponsive("peer never acknowledged".into()),
            SwarmError::SegmentFailoverExhausted("no standby".into()),
            SwarmError::Network("upstream unreachable".into()),
            SwarmError::ServiceUnavailable("worker restarting".into()),
        ] {
            assert_eq!(failure_log_level(&err), FailureLevel::Warn, "{err}");
        }
    }

    /// `failure_log_level` consults `classify_error`, so the classifier must be
    /// pure. It used to `tracing::error!` from its catch-all arm, which would
    /// make merely *asking* what level to use emit an ERROR line of its own —
    /// the precise thing this change removes.
    ///
    /// Asserted behaviourally, by counting emitted events, rather than by
    /// scanning the source for `tracing::` — the first cut did the latter and
    /// tripped over the comment that explains the removal.
    #[test]
    fn classifying_an_error_does_not_log() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Arc;

        #[derive(Clone)]
        struct Counter(Arc<AtomicUsize>);

        impl tracing::subscriber::Subscriber for Counter {
            fn enabled(&self, _: &tracing::Metadata<'_>) -> bool {
                true
            }
            fn new_span(&self, _: &tracing::span::Attributes<'_>) -> tracing::Id {
                tracing::Id::from_u64(1)
            }
            fn record(&self, _: &tracing::Id, _: &tracing::span::Record<'_>) {}
            fn record_follows_from(&self, _: &tracing::Id, _: &tracing::Id) {}
            fn event(&self, _: &tracing::Event<'_>) {
                self.0.fetch_add(1, Ordering::SeqCst);
            }
            fn enter(&self, _: &tracing::Id) {}
            fn exit(&self, _: &tracing::Id) {}
        }

        let count = Arc::new(AtomicUsize::new(0));
        let sub = Counter(count.clone());

        tracing::subscriber::with_default(sub, || {
            // The catch-all arm — the one that used to log — plus a couple of
            // ordinary variants for good measure.
            let _ = classify_error(&SwarmError::InvalidSignature);
            let _ = classify_error(&SwarmError::Validation("too long".into()));
            let _ = classify_error(&SwarmError::Internal("bug".into()));
            // And the level helper itself, which is the actual caller at risk.
            let _ = failure_log_level(&SwarmError::InvalidSignature);
        });

        assert_eq!(
            count.load(Ordering::SeqCst),
            0,
            "classify_error must stay free of logging side effects — a logging \
             path has to be able to call it without emitting a line of its own"
        );
    }

    /// The caller's own mistake must never be reported as this server failing.
    ///
    /// The streaming encoders could not reach this classification, so they
    /// hardcoded `server_error` for every failure: an over-long prompt was a
    /// `400 invalid_request_error` when the client did not stream and a
    /// `server_error` when it did (measured on the released v0.3.95 binary,
    /// 2026-08-12). Both now read the type from here, so the two surfaces agree
    /// by construction — this test pins the classification they share.
    #[test]
    fn a_users_own_input_error_is_never_classified_as_a_server_fault() {
        for err in [
            SwarmError::Validation("This conversation is too long".into()),
            SwarmError::InvalidNickname("bad".into()),
        ] {
            let (status, _msg, error_type) = classify_error(&err);
            assert_eq!(status, StatusCode::BAD_REQUEST, "{err}");
            assert_eq!(error_type, "invalid_request_error", "{err}");
            assert!(
                !status.is_server_error(),
                "a caller-fixable error must not be a 5xx: {err}"
            );
        }
    }

    /// A peer's diagnosis of the CALLER's mistake must not arrive as a server
    /// fault.
    ///
    /// `SwarmError` survives neither the worker hop nor the network hop. Only
    /// the worker one recovered the class, so an over-long prompt sent to a
    /// model held by peers came back `500 server_error` while the identical
    /// request on a local model came back `400 invalid_request_error`
    /// (measured 2026-08-12). It also charged the peer, which had done nothing
    /// wrong — `failure_is_penalty_worthy` exempts `Validation` but never saw
    /// one, because the class had already been flattened to `Inference`.
    #[test]
    fn a_peers_diagnosis_of_the_callers_mistake_survives_the_wire() {
        let from_peer = "Inference error: Validation error: This conversation is too long \
                         for qwen2.5-0.5b-instruct-fp16: 9020 tokens of prompt plus 20 \
                         reserved for the reply is 9040, and the model's limit is 4096.";
        let recovered = reclassify_flattened_error(from_peer).expect("class must be recovered");
        let (status, _msg, error_type) = classify_error(&recovered);
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(error_type, "invalid_request_error");
        assert!(
            matches!(recovered, SwarmError::Validation(ref d) if d.starts_with("This conversation"))
        );
    }

    /// A genuine fault must NOT be laundered into something caller-fixable —
    /// that would hide a real bug behind a 400 and stop it being retried.
    #[test]
    fn an_ordinary_failure_is_left_as_a_server_fault() {
        for msg in [
            "Inference error: tensor shape mismatch",
            "CUDA out of memory",
            "Validation error: ",
        ] {
            assert!(
                reclassify_flattened_error(msg).is_none(),
                "{msg:?} must not be reclassified"
            );
        }
    }

    /// A permanent capability gap must not wear a "try again later" status.
    ///
    /// Embeddings returned `503 server_error` for something no amount of
    /// waiting can fix — inference runs in worker subprocesses in every
    /// supported configuration — so a client with retry-on-5xx re-sends it
    /// forever and monitoring reads a missing feature as this node being
    /// unwell (measured 2026-08-15). 501 is the honest status, and it is the
    /// same lesson as #295 expressed in the status rather than the hint.
    #[test]
    fn an_unimplemented_feature_is_501_not_a_retryable_503() {
        let err = SwarmError::NotImplemented("no embeddings in this build".into());
        let (status, _msg, error_type) = classify_error(&err);
        assert_eq!(status, StatusCode::NOT_IMPLEMENTED);
        assert_eq!(error_type, "not_implemented_error");
        assert_ne!(
            status,
            StatusCode::SERVICE_UNAVAILABLE,
            "503 invites a retry that can never succeed"
        );
        // And it must never be blamed on a peer.
        assert!(error_hint(&err).is_none() || !error_hint(&err).unwrap().contains("Try again"));
    }

    /// A policy refusal is this node declining by configuration, not a crash.
    /// It is the case that reached an Anthropic streaming client as a clean
    /// empty turn until 2026-08-12, so the classification it now streams with
    /// is worth pinning.
    #[test]
    fn a_policy_refusal_classifies_as_unavailable_not_internal() {
        let err = SwarmError::PromptPrivacyUnavailable {
            model_id: "m".into(),
        };
        let (status, _msg, error_type) = classify_error(&err);
        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(error_type, "prompt_privacy_error");
    }

    #[test]
    fn hint_for_no_model_loaded() {
        let err = SwarmError::NoModelLoaded;
        assert!(error_hint(&err).unwrap().contains("dashboard"));
    }

    #[test]
    fn hint_for_insufficient_credits() {
        let err = SwarmError::InsufficientCredits {
            balance: 10,
            required: 100,
        };
        assert!(error_hint(&err).unwrap().contains("Earn credits"));
    }

    #[test]
    fn hint_for_unauthorized() {
        let err = SwarmError::Unauthorized("missing token".into());
        assert!(error_hint(&err).unwrap().contains("Authorization"));
    }

    /// A caller who already sent a valid API key must not be told to go and
    /// find their API key. The update endpoints are loopback-only on purpose,
    /// but they filed that refusal under `Unauthorized`, so a LAN or Docker
    /// admin clicking "Check for updates" got a 401 whose hint pointed at the
    /// one thing that was never wrong — the same shape as gotcha #295.
    ///
    /// Asserted on the ADVICE rather than the wording: pinning the sentence
    /// lets a rewrite keep passing while the user is still looping.
    #[test]
    fn a_local_only_refusal_never_blames_the_callers_api_key() {
        let err = SwarmError::LocalOnly("Applying an update".into());
        let hint = error_hint(&err).expect("a refusal the caller cannot retry needs a hint");
        let misleading = ["api key can be found", "bearer", "authorization header"];
        for phrase in misleading {
            assert!(
                !hint.to_lowercase().contains(phrase),
                "hint sends an authenticated caller after credentials: {hint}"
            );
        }
        // And it must say where the action CAN be performed, or the user is
        // left knowing only that it failed.
        assert!(hint.to_lowercase().contains("computer"), "hint: {hint}");

        // The genuinely-unauthenticated case keeps the opposite advice.
        let missing = SwarmError::Unauthorized("missing token".into());
        assert!(error_hint(&missing).unwrap().contains("Authorization"));
    }

    /// 401 says "your credentials were rejected"; these callers authenticated
    /// fine and are being refused on where the request came from.
    #[test]
    fn a_local_only_refusal_is_forbidden_not_unauthenticated() {
        let (status, _, error_type) = classify_error(&SwarmError::LocalOnly("Shutdown".into()));
        assert_eq!(status, StatusCode::FORBIDDEN);
        assert_eq!(error_type, "permission_error");
        let (status, _, _) = classify_error(&SwarmError::Unauthorized("no key".into()));
        assert_eq!(status, StatusCode::UNAUTHORIZED, "regression guard");
    }

    #[test]
    fn hint_for_insufficient_disk() {
        let err = SwarmError::InsufficientDisk {
            need_mb: 1000,
            have_mb: 100,
        };
        assert!(error_hint(&err).unwrap().contains("max_disk_mb"));
    }

    #[test]
    fn hint_for_shard_integrity() {
        let err = SwarmError::ShardIntegrity {
            expected: "abc".into(),
            actual: "def".into(),
        };
        assert!(error_hint(&err).unwrap().contains("corrupted"));
    }

    /// A failure that retrying cannot fix must never be answered with advice to
    /// retry. This is the third time that rule has been broken the same way: the
    /// generic `PipelineError` hint asserts a transient cause ("a peer went
    /// offline"), and each permanent failure filed under that variant inherited
    /// it — first the missing-layer case (reported 2026-07-30), then prompt
    /// privacy, which is why the latter is now its own variant.
    ///
    /// Asserting on the ADVICE rather than the wording is deliberate: the prose
    /// gets rewritten, and a test pinned to a phrase would pass while the user
    /// was still being sent round a loop with no exit.
    #[test]
    fn permanent_failures_are_never_told_to_retry() {
        for err in [
            SwarmError::PromptPrivacyUnavailable {
                model_id: "m".into(),
            },
            SwarmError::PipelineError("No node available for layer 10".into()),
        ] {
            let hint = error_hint(&err).expect("permanent failure needs a hint");
            let lower = hint.to_lowercase();
            assert!(
                !lower.contains("try again")
                    && !lower.contains("a different route")
                    && !lower.contains("peer went offline"),
                "{err} is permanent but its hint advises retrying: {hint}"
            );
        }
    }

    /// Prompt privacy is a policy refusal by THIS node, not a bug in it, and not
    /// the caller's mistake — so it answers 503, the same as the sibling
    /// `PrivateModeUnavailable`. It answered 500 "server_error" until
    /// 2026-08-11, which reports a deliberate configuration as a crash.
    #[test]
    fn prompt_privacy_refusal_is_service_unavailable_not_a_bug() {
        let resp = ApiError(SwarmError::PromptPrivacyUnavailable {
            model_id: "m".into(),
        })
        .into_response();
        assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
    }

    /// A peer going silent mid-request is a transient serve failure, not a bug
    /// in this node: 503, not 500 — it wore `PipelineError`'s 500 until
    /// 2026-08-16, telling monitoring the node the user talked to was broken.
    /// The variant must also survive the typeless boundaries (worker IPC, the
    /// wire), or a nested route re-flattens it back into a 500.
    #[test]
    fn a_silent_peer_is_service_unavailable_and_survives_flattening() {
        let err = SwarmError::PeerUnresponsive(
            "remote-generate: peer never acknowledged request_id=x (silent drop or disconnect)"
                .into(),
        );
        let (status, _, _) = classify_error(&err);
        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);

        let flattened = format!("Inference error: {err}");
        match reclassify_flattened_error(&flattened) {
            Some(SwarmError::PeerUnresponsive(d)) => {
                assert!(d.contains("never acknowledged"))
            }
            other => panic!("expected PeerUnresponsive back, got {other:?}"),
        }

        // The hint may promise the retry branch — the cause is KNOWN to be
        // transient — but it must actually advise retrying.
        let hint = error_hint(&err).expect("transient peer failure needs a hint");
        assert!(
            hint.to_lowercase().contains("try again"),
            "hint must advise the retry that helps: {hint}"
        );
    }

    /// The second 500-shaped transient from the same report: a mid-pipeline
    /// holder failure with no standby. Same contract — 503, survives
    /// flattening, hint advises the retry.
    #[test]
    fn exhausted_failover_is_service_unavailable_and_survives_flattening() {
        let err = SwarmError::SegmentFailoverExhausted(
            "Segment 1 failed with no standby available".into(),
        );
        let (status, _, _) = classify_error(&err);
        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);

        let flattened = format!("Inference error: {err}");
        assert!(matches!(
            reclassify_flattened_error(&flattened),
            Some(SwarmError::SegmentFailoverExhausted(_))
        ));

        let hint = error_hint(&err).expect("transient failure needs a hint");
        assert!(hint.to_lowercase().contains("try again"), "{hint}");
    }

    /// The generic pipeline hint covers causes we cannot tell apart, so it must
    /// not state one of them as fact. It may offer retrying as a possibility;
    /// it may not promise it will work.
    #[test]
    fn generic_pipeline_hint_does_not_assert_a_cause_it_cannot_know() {
        let hint = error_hint(&SwarmError::PipelineError("segment 1 failed".into())).unwrap();
        let lower = hint.to_lowercase();
        assert!(
            !lower.contains("this usually means") && !lower.contains("a different route will"),
            "generic hint asserts a cause it cannot know: {hint}"
        );
    }

    /// The swarm not having all of a model is a capacity shortfall, not a fault
    /// in the node the user is talking to. Its sibling `InsufficientCapacity` —
    /// same situation, nothing rather than something available — has always
    /// answered 503; this one answered 500 until 2026-08-11, so which status you
    /// got depended on HOW MUCH of the model was missing.
    #[test]
    fn a_model_the_swarm_cannot_complete_is_503_like_its_sibling() {
        let incomplete = ApiError(SwarmError::ModelIncompleteInSwarm {
            model_id: "m".into(),
            layer: 0,
        })
        .into_response();
        let nothing_available =
            ApiError(SwarmError::InsufficientCapacity(ModelId("m".into()))).into_response();
        assert_eq!(
            incomplete.status(),
            nothing_available.status(),
            "missing SOME of a model and missing ALL of it must not differ in status"
        );
        assert_eq!(incomplete.status(), StatusCode::SERVICE_UNAVAILABLE);
    }

    /// Every refusal a user can hit must say what to do about it. Private mode
    /// was the only one with no hint at all, and its message ("missing shards:
    /// [0, 1, 2, 3]") is the least readable of them — a non-technical user was
    /// shown a refusal, a list of numbers, and no way forward.
    ///
    /// The hint must not push them to disable private mode without saying what
    /// that costs: it is the fastest way to clear the error and the only one
    /// that changes where their prompts travel.
    #[test]
    fn private_mode_refusal_explains_the_options_including_the_cost() {
        let hint = error_hint(&SwarmError::PrivateModeUnavailable {
            model_id: "m".into(),
            missing_shards: vec![0, 1],
        })
        .expect("private mode refusal needs a hint like every other refusal");
        let lower = hint.to_lowercase();
        assert!(
            lower.contains("get-model") || lower.contains("pool"),
            "must name a way to keep private mode on, got {hint:?}"
        );
        assert!(
            lower.contains("other people's machines") || lower.contains("sent to other"),
            "if it offers turning private mode off, it must say what that costs: {hint:?}"
        );
    }

    #[test]
    fn no_hint_for_generic_error() {
        let err = SwarmError::Internal("something broke".into());
        assert!(error_hint(&err).is_none());
    }
}
