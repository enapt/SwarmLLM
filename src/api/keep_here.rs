//! A caller's instruction to keep ONE request on this computer (FUTURE_WORK
//! #251 (2b)).
//!
//! The owner can keep a whole model here (its ⚙ panel) or a whole node
//! (private mode). This is the same choice for one request, made by whoever
//! sends it — a script, an SDK, Claude Code — on every surface:
//!
//! - the header `X-SwarmLLM-Keep-On-This-Computer: true`, which reaches Claude
//!   Code through `ANTHROPIC_CUSTOM_HEADERS` and any OpenAI SDK through its
//!   default-headers option;
//! - the body field `"swarmllm": {"keep_on_this_computer": true}` (an SDK's
//!   `extra_body`), namespaced the way OpenRouter namespaces its `provider`
//!   object so it cannot collide with a field OpenAI adds later;
//! - on MCP, `params._meta["com.swarmllm/keep_on_this_computer"]` (the
//!   reverse-DNS key form MCP reserves for implementations), or the header on
//!   the `/mcp` request.
//!
//! **The strictest instruction wins**, as OpenRouter ORs a request's `zdr`
//! with the account's: `true` anywhere keeps the request here, and `false`
//! never lifts the owner's setting or private mode. And it **fails closed**: a
//! value this node cannot read is a 400, never silently ignored — a privacy
//! instruction that was dropped would let the request out while the caller
//! believes it stayed.

use axum::http::HeaderMap;
use serde::{Deserialize, Serialize};

use crate::error::SwarmError;
use crate::types::KeepAsked;

/// The request header (HTTP header names are case-insensitive).
pub const HEADER: &str = "x-swarmllm-keep-on-this-computer";

/// The key under an MCP request's `params._meta`.
pub const MCP_META_KEY: &str = "com.swarmllm/keep_on_this_computer";

/// The `swarmllm` object a request body may carry. Unknown keys are refused,
/// so a misspelt `keep_on_this_computr` is an error rather than a request that
/// quietly left this computer.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct SwarmllmOptions {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub keep_on_this_computer: Option<bool>,
}

impl SwarmllmOptions {
    /// This body's own instruction.
    pub fn asked(&self) -> KeepAsked {
        KeepAsked::kept_if(self.keep_on_this_computer == Some(true))
    }
}

/// What the header says, if it is sent: `true`/`false` (also `1`/`0`,
/// `yes`/`no`, any case). Several values keep the request here if any says
/// so; one this node cannot read is a 400.
pub fn from_header(headers: &HeaderMap) -> Result<KeepAsked, SwarmError> {
    let mut keep = false;
    for value in headers.get_all(HEADER) {
        let text = value.to_str().unwrap_or("").trim().to_ascii_lowercase();
        keep |= match text.as_str() {
            "true" | "1" | "yes" => true,
            "false" | "0" | "no" => false,
            _ => {
                return Err(SwarmError::Validation(format!(
                    "the X-SwarmLLM-Keep-On-This-Computer header must be true or false, not {:?}",
                    value.to_str().unwrap_or("<not text>")
                )))
            }
        };
    }
    Ok(KeepAsked::kept_if(keep))
}

/// The instruction a request carries in its header and its body, the strictest
/// of the two.
pub fn asked(headers: &HeaderMap, body: Option<&SwarmllmOptions>) -> Result<KeepAsked, SwarmError> {
    Ok(strictest(
        from_header(headers)?,
        body.map_or(KeepAsked::Unasked, SwarmllmOptions::asked),
    ))
}

/// The stricter of two instructions.
pub fn strictest(a: KeepAsked, b: KeepAsked) -> KeepAsked {
    KeepAsked::kept_if(a.is_kept() || b.is_kept())
}

/// What an MCP request's `params._meta` says: `true` keeps it here, `false`
/// or absent does not, anything else is refused — and so is a `_meta` that is
/// not an object, which would otherwise read as "no instruction" (MCP
/// defines `_meta` as an object). A key repeated inside `_meta` cannot be
/// seen here: the JSON was parsed into a map that kept the last value.
pub fn from_mcp_meta(params: Option<&serde_json::Value>) -> Result<KeepAsked, SwarmError> {
    let meta = match params.and_then(|p| p.get("_meta")) {
        None | Some(serde_json::Value::Null) => return Ok(KeepAsked::Unasked),
        Some(serde_json::Value::Object(meta)) => meta,
        Some(other) => {
            return Err(SwarmError::Validation(format!(
                "_meta must be an object, not {other}"
            )))
        }
    };
    match meta.get(MCP_META_KEY) {
        None | Some(serde_json::Value::Bool(false)) | Some(serde_json::Value::Null) => {
            Ok(KeepAsked::Unasked)
        }
        Some(serde_json::Value::Bool(true)) => Ok(KeepAsked::OnThisComputer),
        Some(other) => Err(SwarmError::Validation(format!(
            "_meta[\"{MCP_META_KEY}\"] must be true or false, not {other}"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn with(values: &[&str]) -> HeaderMap {
        let mut headers = HeaderMap::new();
        for v in values {
            headers.append(HEADER, v.parse().unwrap());
        }
        headers
    }

    /// The strictest instruction wins and nothing loosens: `true` in either
    /// place keeps the request, `false` in the other does not undo it.
    #[test]
    fn the_strictest_instruction_wins() {
        let keep = SwarmllmOptions {
            keep_on_this_computer: Some(true),
        };
        let loose = SwarmllmOptions {
            keep_on_this_computer: Some(false),
        };
        assert_eq!(asked(&with(&[]), None).unwrap(), KeepAsked::Unasked);
        assert_eq!(
            asked(&with(&["TRUE"]), None).unwrap(),
            KeepAsked::OnThisComputer
        );
        assert_eq!(
            asked(&with(&["false"]), Some(&keep)).unwrap(),
            KeepAsked::OnThisComputer
        );
        assert_eq!(
            asked(&with(&["yes"]), Some(&loose)).unwrap(),
            KeepAsked::OnThisComputer
        );
        assert_eq!(
            asked(&with(&["0", "1"]), None).unwrap(),
            KeepAsked::OnThisComputer
        );
        assert_eq!(
            asked(&with(&["no"]), Some(&loose)).unwrap(),
            KeepAsked::Unasked
        );
    }

    /// A privacy instruction this node cannot read is refused, never dropped:
    /// a bad header value, a misspelt body key, a non-boolean `_meta` value, a
    /// `_meta` that is not an object.
    #[test]
    fn an_instruction_that_cannot_be_read_is_refused() {
        assert!(matches!(
            asked(&with(&["please"]), None),
            Err(SwarmError::Validation(_))
        ));
        assert!(
            serde_json::from_str::<SwarmllmOptions>(r#"{"keep_on_this_computr": true}"#).is_err()
        );
        let meta = |v: serde_json::Value| serde_json::json!({ "_meta": { MCP_META_KEY: v } });
        assert_eq!(
            from_mcp_meta(Some(&meta(true.into()))).unwrap(),
            KeepAsked::OnThisComputer
        );
        assert_eq!(
            from_mcp_meta(Some(&meta(false.into()))).unwrap(),
            KeepAsked::Unasked
        );
        assert_eq!(from_mcp_meta(None).unwrap(), KeepAsked::Unasked);
        assert!(from_mcp_meta(Some(&meta("yes".into()))).is_err());
        // A `_meta` that is not an object would hide the key entirely.
        for not_an_object in [serde_json::json!("keep"), serde_json::json!([true])] {
            assert!(from_mcp_meta(Some(&serde_json::json!({ "_meta": not_an_object }))).is_err());
        }
    }
}
