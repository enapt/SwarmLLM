//! Shared dispatch helpers used by the compare/research/batch_prompts tools.
//!
//! Each of these tools fans out the same HTTP call to N models and collects
//! results back. Helpers live here so tools.rs (or mod.rs) stays focused on
//! per-tool shape + parameter validation.

use serde_json::json;

/// Per-task timeout for MCP multi-model calls (matches tool_chat's 120s).
pub(super) const MCP_TASK_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(120);

/// Count results that both succeeded AND carry an answer.
///
/// **"Did not error" is not the same as "answered".** A model can return a
/// perfectly successful response with empty content — a reasoning model that
/// spent its whole token budget on a hidden thinking trace, or a reply whose
/// every token was stripped as a control-token artifact. The tools that
/// summarise a fan-out counted `status == "ok"` and called it success, so a
/// batch reported `tasks_completed: 5` of 6 when one of those five was blank,
/// and the caller had no way to tell which (reported 2026-08-10, alongside a
/// cloud model that burned its full 32-token budget and returned nothing).
///
/// **This reads the `empty` flag stamped by `spawn_model_call_task`; it does
/// NOT go looking for the answer text itself.** The three tools deliberately
/// name that text differently on the wire — `content` for compare and batch,
/// `response` for research — so a helper that inspected the payload would have
/// to know every one of those names, and would silently mis-report the moment
/// a fourth tool chose a new one. That is not hypothetical: the first cut of
/// this checked `content` only, which flagged *every* successful research
/// answer as blank and would have reported `responses_with_content: 0` on a
/// tool whose entire job is answering questions. Deciding blankness at the
/// choke point, where the text is and before any tool names it, is what makes
/// the key irrelevant.
pub(super) fn count_answered(results: &[serde_json::Value]) -> usize {
    results
        .iter()
        .filter(|r| r["status"] == "ok" && r["empty"] != serde_json::Value::Bool(true))
        .count()
}

/// Await every spawned task, bounded by ONE deadline for the whole set.
///
/// **The deadline is shared deliberately.** This used to start a fresh
/// `MCP_TASK_TIMEOUT` for each handle and await them in order, so the constant
/// that reads like a bound was per-task: twenty tasks could legitimately run
/// for twenty times it — forty minutes against a number documented as two.
/// The tasks are already running CONCURRENTLY (they are spawned before this is
/// called), so the batch should finish in about the slowest one's time, and a
/// single deadline is what actually expresses that.
///
/// It matters beyond tidiness. Reported 2026-08-10: a six-task batch timed out
/// on the CLIENT with no results returned, while the server went on executing
/// and billing for it — "you can pay for a batch and never see any of the
/// output". A server that can overrun its own stated bound by the number of
/// tasks is how that happens. Bounded here, the whole call returns within
/// `MCP_TASK_TIMEOUT` whatever the batch size, with the unfinished tasks
/// reported as timed out rather than silently dropped.
pub(super) async fn collect_handle_results(
    handles: Vec<tokio::task::JoinHandle<serde_json::Value>>,
) -> Vec<serde_json::Value> {
    collect_within(MCP_TASK_TIMEOUT, handles).await
}

/// `collect_handle_results` with the budget supplied, so the shared-deadline
/// behaviour can be tested against a short one instead of the real two minutes.
async fn collect_within(
    budget: std::time::Duration,
    handles: Vec<tokio::task::JoinHandle<serde_json::Value>>,
) -> Vec<serde_json::Value> {
    let deadline = tokio::time::Instant::now() + budget;
    let mut results = Vec::with_capacity(handles.len());
    for mut handle in handles {
        // `&mut handle` so the handle survives the timeout and can be aborted.
        match tokio::time::timeout_at(deadline, &mut handle).await {
            Ok(Ok(result)) => results.push(result),
            Ok(Err(e)) => {
                results.push(json!({"error": format!("Task failed: {e}"), "status": "error"}))
            }
            Err(_) => {
                // Abort rather than leave it running: the caller has been told
                // this task timed out, so letting it finish later would bill
                // for work nobody will ever see.
                handle.abort();
                results.push(json!({
                    "error": format!("Request timed out ({}s for the batch)", budget.as_secs()),
                    "status": "error",
                }))
            }
        }
    }
    results
}

/// Extract text content and token usage from an Anthropic Messages API response body.
pub(super) fn extract_anthropic_response(body: &serde_json::Value) -> (String, u64, u64) {
    let content = body["content"]
        .as_array()
        .and_then(|arr| {
            arr.iter()
                .filter_map(|b| b.get("text").and_then(|t| t.as_str()))
                .next()
        })
        .unwrap_or("")
        .to_string();
    let input_tokens = body["usage"]["input_tokens"].as_u64().unwrap_or(0);
    let output_tokens = body["usage"]["output_tokens"].as_u64().unwrap_or(0);
    (content, input_tokens, output_tokens)
}

/// Result of a single model dispatch call used by MCP compare/research/batch tools.
pub(super) struct ModelCallResult {
    pub content: String,
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub elapsed_ms: u64,
    /// None on success, Some(message) on error.
    pub error: Option<String>,
}

/// Send a prompt to a model endpoint and return the parsed result.
///
/// Shared core for tool_compare, tool_research, and tool_batch_prompts.
#[allow(clippy::too_many_arguments)]
pub(super) async fn dispatch_model_call(
    client: &reqwest::Client,
    url: &str,
    api_key: &str,
    model_id: &str,
    prompt: &str,
    system: Option<&str>,
    temperature: f32,
    max_tokens: u32,
) -> ModelCallResult {
    let start = std::time::Instant::now();

    let mut body = json!({
        "model": model_id,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": [{"role": "user", "content": prompt}],
        "stream": false,
    });
    if let Some(sys) = system {
        body["system"] = json!(sys);
    }

    let result = client
        .post(url)
        .header("Authorization", format!("Bearer {api_key}"))
        .header("Content-Type", "application/json")
        .json(&body)
        .send()
        .await;

    let elapsed_ms = start.elapsed().as_millis() as u64;

    match result {
        Ok(resp) if resp.status().is_success() => {
            let resp_body: serde_json::Value = resp
                .json()
                .await
                .unwrap_or(json!({"error": "parse failed"}));
            let (content, input_tokens, output_tokens) = extract_anthropic_response(&resp_body);
            ModelCallResult {
                content,
                input_tokens,
                output_tokens,
                elapsed_ms,
                error: None,
            }
        }
        Ok(resp) => {
            let status = resp.status().as_u16();
            let body = resp.text().await.unwrap_or_default();
            let truncated = crate::api::scrub_truncate_error(&body);
            ModelCallResult {
                content: String::new(),
                input_tokens: 0,
                output_tokens: 0,
                elapsed_ms,
                error: Some(format!("HTTP {status}: {truncated}")),
            }
        }
        Err(e) => ModelCallResult {
            content: String::new(),
            input_tokens: 0,
            output_tokens: 0,
            elapsed_ms,
            error: Some(format!("{e}")),
        },
    }
}

/// Spawn a single model dispatch call as a detached task, with caller-supplied
/// JSON shaping applied to the result. Shared plumbing for tool_compare,
/// tool_research, and tool_batch_prompts, each of which uses slightly different
/// output keys (`content` vs `response`, with or without `task_id`).
#[allow(clippy::too_many_arguments)]
pub(super) fn spawn_model_call_task<F>(
    client: reqwest::Client,
    base_url: &str,
    api_key: String,
    model_id: String,
    prompt: String,
    system: Option<String>,
    temperature: f32,
    max_tokens: u32,
    shape: F,
) -> tokio::task::JoinHandle<serde_json::Value>
where
    F: FnOnce(&str, ModelCallResult) -> serde_json::Value + Send + 'static,
{
    let url = format!("{base_url}/v1/messages");
    tokio::spawn(async move {
        let r = dispatch_model_call(
            &client,
            &url,
            &api_key,
            &model_id,
            &prompt,
            system.as_deref(),
            temperature,
            max_tokens,
        )
        .await;
        // Whether the model actually said anything is decided HERE — where the
        // text still is, and before the per-tool closure gets to name it. Every
        // real model call in every fan-out tool passes through this function,
        // so a new tool inherits the flag with no author action and cannot get
        // it wrong by picking a different key for its answer field. See
        // `count_answered` for what went wrong when this was done downstream.
        let blank = r.error.is_none() && r.content.trim().is_empty();
        let mut shaped = shape(&model_id, r);
        stamp_blank_answer(&mut shaped, blank);
        shaped
    })
}

/// Stamp the "succeeded but said nothing" marker onto an already-shaped result.
///
/// Split out from `spawn_model_call_task` so the marker's behaviour is testable
/// without an HTTP round trip — the property that matters (it does not care what
/// the tool called its answer field) is otherwise only reachable through a live
/// model call.
pub(super) fn stamp_blank_answer(shaped: &mut serde_json::Value, blank: bool) {
    if !blank {
        return;
    }
    // An explicit flag rather than a changed `status`: clients already branch
    // on "ok", and silently reclassifying a success would break them in order
    // to fix a reporting gap.
    if let Some(obj) = shaped.as_object_mut() {
        obj.insert("empty".to_string(), serde_json::Value::Bool(true));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    /// The three fan-out tools shape a successful answer differently on the
    /// wire. These are the real shapes, kept here so a tool that changes its
    /// key has to look at this test.
    fn compare_shaped(text: &str) -> Value {
        json!({"model": "m", "content": text, "status": "ok"})
    }
    fn research_shaped(text: &str) -> Value {
        json!({"model": "m", "response": text, "status": "ok"})
    }
    fn batch_shaped(text: &str) -> Value {
        json!({"task_id": "t", "model": "m", "content": text, "status": "ok"})
    }

    /// Reproduce what `spawn_model_call_task` does for one successful call: the
    /// blankness verdict is taken from the model's TEXT, then the tool's own
    /// shape is stamped with it. Going through this rather than asserting on a
    /// hand-built flag is what makes these tests able to fail — an earlier
    /// version built the shapes and only exercised `count_answered`, so it
    /// passed just as happily with the broken logic in place.
    fn served(shape: fn(&str) -> Value, text: &str) -> Value {
        let mut v = shape(text);
        stamp_blank_answer(&mut v, text.trim().is_empty());
        v
    }

    #[test]
    fn blank_answers_are_flagged_whatever_the_tool_named_the_field() {
        for shaped in [
            served(compare_shaped, ""),
            served(research_shaped, ""),
            served(batch_shaped, "   \n "),
        ] {
            assert_eq!(shaped["empty"], json!(true), "not flagged: {shaped}");
            assert_eq!(shaped["status"], json!("ok"), "status must not change");
            assert_eq!(count_answered(std::slice::from_ref(&shaped)), 0);
        }
    }

    #[test]
    fn real_answers_are_counted_whatever_the_tool_named_the_field() {
        // The regression this pins: an earlier cut decided blankness downstream
        // by reading `content`, so research — which calls the same text
        // `response` — had every one of its successful answers flagged empty
        // and would have reported `responses_with_content: 0` on the one tool
        // whose entire job is answering questions.
        let results = [
            served(compare_shaped, "hello"),
            served(research_shaped, "hello"),
            served(batch_shaped, "hello"),
        ];
        for r in &results {
            assert!(
                r.get("empty").is_none(),
                "a real answer was flagged empty: {r}"
            );
        }
        assert_eq!(count_answered(&results), 3);
    }

    #[test]
    fn errored_results_are_never_counted_as_answered() {
        let results = [
            json!({"model": "m", "error": "boom", "status": "error"}),
            compare_shaped("hi"),
        ];
        assert_eq!(count_answered(&results), 1);
    }

    #[test]
    fn stamp_is_a_no_op_when_the_answer_has_content() {
        let mut shaped = compare_shaped("hi");
        stamp_blank_answer(&mut shaped, false);
        assert!(shaped.get("empty").is_none());
    }

    const TEST_BUDGET: std::time::Duration = std::time::Duration::from_millis(500);

    fn never_finishes() -> tokio::task::JoinHandle<Value> {
        tokio::spawn(async {
            tokio::time::sleep(std::time::Duration::from_secs(86_400)).await;
            json!({"status": "ok"})
        })
    }

    #[tokio::test]
    async fn the_batch_deadline_is_shared_not_per_task() {
        let start = std::time::Instant::now();
        // Three tasks that never finish. A per-task timeout awaited in a loop
        // serialises into 3 x the budget; one shared deadline bounds the set.
        let handles: Vec<_> = (0..3).map(|_| never_finishes()).collect();

        let results = collect_within(TEST_BUDGET, handles).await;
        let elapsed = start.elapsed();

        assert_eq!(results.len(), 3, "every task must be reported");
        assert!(results.iter().all(|r| r["status"] == "error"));
        assert!(
            elapsed < TEST_BUDGET * 2,
            "batch took {elapsed:?} against a {TEST_BUDGET:?} budget — the \
             timeout is being applied per task again"
        );
    }

    #[tokio::test]
    async fn results_keep_their_input_order_when_some_time_out() {
        let quick = tokio::spawn(async { json!({"model": "quick", "status": "ok"}) });
        let never = never_finishes();
        let quick2 = tokio::spawn(async { json!({"model": "quick2", "status": "ok"}) });

        // The slow task sits in the middle: callers correlate results back to
        // their input list positionally, so a timeout must occupy its slot
        // rather than be dropped.
        let results = collect_within(TEST_BUDGET, vec![quick, never, quick2]).await;

        assert_eq!(results.len(), 3);
        assert_eq!(results[0]["model"], json!("quick"));
        assert_eq!(results[1]["status"], json!("error"));
        assert_eq!(results[2]["model"], json!("quick2"));
    }
}
