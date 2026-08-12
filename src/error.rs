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
            // Log the full error internally but return a generic message
            // to avoid leaking internal paths, peer errors, or DB details.
            tracing::error!(
                error = %err,
                "Internal server error"
            );
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                "An internal error occurred".to_string(),
                "server_error",
            )
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> axum::response::Response {
        let (status, message, error_type) = classify_error(&self.0);
        // Log 5xx errors so they appear in tracing output (catch non-catch-all 5xx)
        if status.is_server_error() && error_type != "server_error" {
            tracing::error!(
                status = status.as_u16(),
                error = %message,
                "Server error"
            );
        }

        let hint = error_hint(&self.0);

        let mut error_obj = serde_json::json!({
            "message": message,
            "type": error_type,
            "param": null,
            "code": error_type
        });
        if let Some(hint_text) = hint {
            error_obj["hint"] = serde_json::Value::String(hint_text.to_string());
        }

        (status, Json(serde_json::json!({ "error": error_obj }))).into_response()
    }
}

/// Return an actionable hint for common error variants.
/// These are user-facing — no curl commands or API paths.
pub fn error_hint(err: &SwarmError) -> Option<&'static str> {
    match err {
        SwarmError::ModelNotAvailable(_) => Some(
            "This model isn't available yet. Open the Models tab in the dashboard to browse \
             and download models, or wait for auto-manage to acquire it from the network.",
        ),
        SwarmError::NoModelLoaded => Some(
            "No model is loaded yet. Go to the dashboard and select a model to download, \
             or connect to more peers so models can be served from the network.",
        ),
        SwarmError::InsufficientCredits { .. } => Some(
            "Your credit balance is too low. Earn credits by hosting model shards \
             (happens automatically) or serving inference for other users. \
             Check your balance on the dashboard.",
        ),
        SwarmError::InsufficientCapacity(_) => Some(
            "Not enough peers have the shards needed for this model. \
             Try again later as more peers come online, or download the model \
             shards yourself from the Models tab.",
        ),
        SwarmError::InsufficientDisk { .. } => Some(
            "Not enough disk space. Free up space or increase the storage limit \
             in Settings → Advanced → max_disk_mb.",
        ),
        SwarmError::Unauthorized(_) => Some(
            "Authentication required. Your API key can be found on the dashboard \
             Settings page. Include it as a Bearer token in the Authorization header.",
        ),
        SwarmError::PeerNotFound(_) => Some(
            "That peer is offline or unreachable. Check your internet connection \
             and try again later.",
        ),
        SwarmError::ShardIntegrity { .. } => Some(
            "A model file was corrupted and will be re-downloaded automatically. \
             This is usually caused by an interrupted download — try again in a moment.",
        ),
        SwarmError::ShardIncomplete { .. } => Some(
            "A model file did not download completely and will be fetched again \
             automatically. This usually means the connection dropped part-way.",
        ),
        // `PipelineError` covers two causes needing OPPOSITE advice, and the
        // generic text asserted the wrong one. A tester was told "a peer went
        // offline mid-request" when the real cause was a piece of the model
        // missing locally that no peer could supply — so following the hint
        // means retrying for ever instead of fetching the piece. Reported
        // 2026-07-30.
        SwarmError::PipelineError(msg) if msg.contains("No node available for layer") => Some(
            "Part of this model isn't available — not on this machine, and not on any \
             connected peer. Fetch the whole model with `swarmllm get-model <name> --all`, \
             or wait for more peers to come online.",
        ),
        // The remaining pipeline failures are a mix of transient and permanent
        // causes, and we do not know which this one is. The old wording picked
        // the transient reading and stated it as fact — "a peer went offline …
        // try again" — so every permanent cause sent the user round a loop with
        // no exit. Say what is known, offer both branches, promise neither.
        SwarmError::PipelineError(_) => Some(
            "The route to run this model couldn't be put together. If a peer dropped out \
             this will fix itself — try once more. If it fails the same way again, the \
             model is missing a piece: fetch it with `swarmllm get-model <name> --all`, \
             or pick a model marked as ready in the dashboard.",
        ),
        SwarmError::ModelIncompleteInSwarm { .. } => Some(
            "Part of this model isn't on any machine that's reachable right now. If a peer \
             just dropped out this may fix itself shortly. Otherwise fetch the model with \
             `swarmllm get-model <name> --all`, or pick a model the dashboard marks as ready.",
        ),
        SwarmError::PromptPrivacyUnavailable { .. } => Some(
            "Prompt privacy keeps your prompt on this machine, which needs the model's \
             first part stored here — and it isn't. Either fetch it with \
             `swarmllm get-model <name>`, or turn prompt privacy off for this model to \
             let the swarm run it. Retrying as-is won't help.",
        ),
        // The only refusal that had no hint, and its message is the least
        // readable of them: "missing shards: [0, 1, 2, 3]" tells a non-technical
        // user nothing, and nothing else on screen says what to do about it.
        //
        // Turning private mode off is deliberately listed LAST and with its
        // consequence stated. It is the quickest way to make the error go away
        // and the only one that changes where the user's prompts travel, so
        // offering it casually would trade someone's privacy for convenience
        // without telling them.
        SwarmError::PrivateModeUnavailable { .. } => Some(
            "Private mode keeps your prompts on your own devices, and none of them has \
             all of this model. Either download it here with `swarmllm get-model <name>`, \
             or add the device that has it to your pool. You can also turn private mode \
             off in Settings — but then your prompts can be sent to other people's \
             machines to run.",
        ),
        SwarmError::InferenceTimeout(_) => Some(
            "The request took too long. Try a shorter prompt, reduce the max tokens, \
             or wait for a less busy time.",
        ),
        SwarmError::Config(_) => Some(
            "There's a configuration issue. Check Settings in the dashboard \
             or review your config.toml file.",
        ),
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
                Some(
                    "Your cloud provider credits may be exhausted or rate-limited. \
                     Top up your account on the provider's website, switch to a free-tier provider \
                     (DeepSeek, Groq, NVIDIA NIM), or use a local swarm model instead.",
                )
            } else if *status == 401 || *status == 403 {
                Some(
                    "Your API key appears to be invalid or revoked. \
                     Update it in Settings → Cloud Providers.",
                )
            } else {
                Some(
                    "The cloud provider returned an error. Try again, \
                     switch to a different model, or use a local swarm model.",
                )
            }
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hint_for_model_not_available() {
        let err = SwarmError::ModelNotAvailable(ModelId("test-model".into()));
        assert!(error_hint(&err).unwrap().contains("Models tab"));
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
