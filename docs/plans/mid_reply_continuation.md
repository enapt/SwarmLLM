# Continuing a reply whose machine failed mid-stream (FUTURE_WORK #236)

**Written 2026-10-08; shipped on main the same day** (FUTURE_WORK #236 closed —
`docs/invariants/scheduling.md` § "A reply whose machine failed mid-stream is
continued" has the rig). The problem: once a streamed reply has started, the
router never re-plans it (`should_retry_after` refuses when `ttft_ms` is set —
a retry from the prompt would show the reader the answer beginning again). So a
peer that refuses, restarts or goes silent mid-reply ends the reply with an
error after part of it was shown (report #005: 71 s in, 24.5 s to the first
token, 46.8 s decoding).

## What others do

- **Petals** keeps each server session's input `history`; on a failure
  `InferenceSession._update_sequence` builds a new route for the failed blocks
  and hands the new session that history, which it replays before the next step
  (`petals/client/inference_session.py`).
- **vLLM** preempts a running sequence under memory pressure and RECOMPUTES its
  cache from its tokens when it is resumed.

Both resume from what was already generated rather than from the prompt. The
coordinator here holds everything such a replay needs: the rendered prompt, and
the exact text the client has been sent.

## The shape

1. **What the client saw is recorded where every token passes.**
   `StreamingTokenTx` (the one sender every emit site uses — it already stamps
   TTFT) appends each event's text to the request's trace. It never enters a
   trace snapshot (reply text is not diagnostics) and is dropped when the
   request ends.
2. **A continuation is a request field the executor honours in ONE place.**
   `InferenceRequest::continuation` (`#[serde(skip)]` — local, not a protocol
   change). `PipelineExecutor::build_prompt_with_header` — the one function
   every path's prompt comes from (local whole model, hand-off, split,
   speculation) — appends the text after the rendered prompt, inside the opened
   assistant turn: the model goes on from where it was. Paths that render
   elsewhere stand aside: a delegated split (the delegate renders the messages
   itself) and the in-process llama executor.
3. **The router continues instead of failing.** In `dispatch_single`'s loop, a
   failure after the first token that a different route could answer — the
   classes `should_retry_after` re-plans, plus a reply that lost tokens in
   transit (continuing from what the client saw is exactly right there) — runs
   again with the continuation and `max_tokens` less what was sent, on the SAME
   token channel, at most `MAX_CONTINUATIONS` times. The machine that failed is
   already barred from the request by whoever produced the error. The output
   reports the whole reply and the tokens of both parts.

## What it does not do

- The batched dispatch path has no re-plan of any kind; it gets none here.
- The text is re-tokenised at the boundary. A word cut mid-token may tokenise
  differently; the model reads it as written and continues — llama.cpp's
  `/completion` continuation works the same way.
- A sampler that reads history (penalties) sees the earlier reply as prompt —
  llama.cpp's `penalty_last_n` window spans prompt and reply alike.
