//! redb-backed persistence for `/v1/responses` records.
//!
//! Tree: `"responses"`. Key: the response `id` (`resp_<hex>`). Value:
//! JSON-serialized `ResponsesRecord`. All I/O goes through the existing
//! `Database::{put_json, get_json, get_all_json, remove}` helpers so the
//! schema stays consistent with the rest of the codebase (single DATA_TABLE,
//! `<tree>/<key>` composite byte keys).
//!
//! Retention is 30 days; the sweep runs from the same
//! `stale_tensor_interval` loop in `daemon/background.rs` that cleans up
//! other time-bounded state.

use serde::{Deserialize, Serialize};

use super::types::{ResponsesRequest, ResponsesResponse};
use crate::error::SwarmError;
use crate::storage::db::Database;

/// 30 days in seconds — matches the OpenAI Responses retention default.
pub const DEFAULT_TTL_SECS: i64 = 30 * 24 * 60 * 60;

/// redb tree name.
pub const TREE: &str = "responses";

/// Persisted form of a single `/v1/responses` exchange.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponsesRecord {
    pub id: String,
    pub created_at: i64,
    pub expires_at: i64,
    pub request: ResponsesRequest,
    pub response: ResponsesResponse,
}

impl ResponsesRecord {
    pub fn new(
        request: ResponsesRequest,
        response: ResponsesResponse,
        now: i64,
        ttl_secs: i64,
    ) -> Self {
        Self {
            id: response.id.clone(),
            created_at: response.created_at,
            expires_at: now + ttl_secs,
            request,
            response,
        }
    }
}

pub fn store(db: &Database, record: &ResponsesRecord) -> Result<(), SwarmError> {
    db.put_json(TREE, &record.id, record)
}

pub fn load(db: &Database, id: &str) -> Result<Option<ResponsesRecord>, SwarmError> {
    db.get_json::<ResponsesRecord>(TREE, id)
}

pub fn delete(db: &Database, id: &str) -> Result<(), SwarmError> {
    db.remove(TREE, id)
}

/// Remove every record whose `expires_at` is at or before `now`.
/// Returns how many were swept.
pub fn sweep_expired(db: &Database, now: i64) -> Result<usize, SwarmError> {
    let all = db.get_all_json::<ResponsesRecord>(TREE)?;
    let mut removed = 0usize;
    for (_, rec) in all {
        if rec.expires_at <= now {
            if let Err(e) = db.remove(TREE, &rec.id) {
                tracing::warn!(id = %rec.id, error = %e, "responses sweep: remove failed");
            } else {
                removed += 1;
            }
        }
    }
    Ok(removed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::openai::responses::types::*;
    use std::collections::HashMap;

    fn tmp_db() -> Database {
        Database::open_temp().expect("temp db")
    }

    fn sample_record(id: &str, expires_at: i64) -> ResponsesRecord {
        let req = ResponsesRequest {
            model: "m".into(),
            input: ResponsesInput::Text("hi".into()),
            instructions: None,
            previous_response_id: None,
            max_output_tokens: None,
            temperature: None,
            top_p: None,
            frequency_penalty: None,
            presence_penalty: None,
            stop: None,
            seed: None,
            user: None,
            metadata: None,
            stream: None,
            store: Some(true),
            background: None,
            parallel_tool_calls: None,
            truncation: None,
            service_tier: None,
            modalities: None,
            include: None,
            tools: None,
            tool_choice: None,
            reasoning: None,
            text: None,
            conversation: None,
            context_management: None,
            extras: HashMap::new(),
        };
        let resp = ResponsesResponse {
            id: id.into(),
            object: "response".into(),
            created_at: 1_700_000_000,
            status: ResponseStatus::Completed,
            model: "m".into(),
            output: Vec::new(),
            output_text: Some(String::new()),
            usage: ResponsesUsage::default(),
            error: None,
            incomplete_details: None,
            previous_response_id: None,
            instructions: None,
            tools: None,
            tool_choice: None,
            parallel_tool_calls: None,
            temperature: Some(0.7),
            top_p: Some(0.9),
            max_output_tokens: Some(2048),
            truncation: None,
            metadata: None,
            user: None,
            reasoning: None,
            text: None,
            modalities: None,
            service_tier: None,
            background: None,
            extras: HashMap::new(),
        };
        ResponsesRecord {
            id: id.into(),
            created_at: resp.created_at,
            expires_at,
            request: req,
            response: resp,
        }
    }

    #[test]
    fn store_load_delete_round_trip() {
        let db = tmp_db();
        let rec = sample_record("resp_abc", 1_700_000_000 + DEFAULT_TTL_SECS);
        store(&db, &rec).unwrap();

        let loaded = load(&db, "resp_abc").unwrap().unwrap();
        assert_eq!(loaded.id, "resp_abc");
        assert_eq!(loaded.response.status, ResponseStatus::Completed);

        delete(&db, "resp_abc").unwrap();
        assert!(load(&db, "resp_abc").unwrap().is_none());
    }

    #[test]
    fn load_missing_returns_none() {
        let db = tmp_db();
        assert!(load(&db, "resp_missing").unwrap().is_none());
    }

    #[test]
    fn sweep_removes_expired_keeps_fresh() {
        let db = tmp_db();
        let now = 2_000_000_000i64;
        store(&db, &sample_record("resp_expired", now - 60)).unwrap();
        store(&db, &sample_record("resp_boundary", now)).unwrap();
        store(&db, &sample_record("resp_fresh", now + DEFAULT_TTL_SECS)).unwrap();

        let removed = sweep_expired(&db, now).unwrap();
        assert_eq!(removed, 2); // expired + boundary (expires_at <= now)

        assert!(load(&db, "resp_expired").unwrap().is_none());
        assert!(load(&db, "resp_boundary").unwrap().is_none());
        assert!(load(&db, "resp_fresh").unwrap().is_some());
    }

    #[test]
    fn sweep_empty_tree_is_ok() {
        let db = tmp_db();
        assert_eq!(sweep_expired(&db, 1_000_000_000).unwrap(), 0);
    }
}
