# Diagnosis Rules

Every entry here was paid for. On 2026-08-03 three wrong causal claims reached
commits in one session, each corrected only after the fact, and a fourth was
caught on the edge. They were not careless — each had a plausible mechanism and
supporting evidence. That is exactly why rules are needed rather than resolve.

The shape is always the same: **a symptom is attributed to the most recent
change, or the most obvious component, before establishing that it does not
happen without it.**

## 0. Look it up before you dig in

Before diagnosing an unfamiliar failure or implementing anything non-trivial,
search for how the problem is understood elsewhere. Most things this project
hits — key rotation, admission control, replay protection, pipeline scheduling,
NAT traversal — are solved problems in systems with more scars than ours. Their
solution usually names the failure mode, and often contains the one detail you
would otherwise get wrong.

**This is not ceremony. Both times it was done on 2026-08-03 it changed the
implementation:**

- **Key rotation.** WireGuard keeps the previous session key *and maintains the
  replay counter per keypair*. That second half is the part that matters: a
  previous-key grace window sharing one replay window would have let the same
  message be accepted twice, once under each key. The design was already heading
  for the grace window; the research is what stopped it being a security
  regression.
- **Headroom routing.** vLLM's approach has a name — Head-Room Admission — and
  reading it surfaced the distinction that mattered: vLLM *preempts and
  recomputes* when saturated because it has nowhere else to go, whereas a swarm
  can route to a peer. That framed the whole design.

Where it was skipped the same day, the work went worse. Time spent was minutes
in both cases.

**What to search for**, in rough order of usefulness:

1. The failure mode as another system names it ("rekey window", "admission
   control", "head-of-line blocking") rather than our symptom wording.
2. How a comparable system solves it, and *why* — the rationale is what
   transfers, since our constraints differ.
3. Known traps in that solution. The per-keypair replay counter is exactly this
   kind of finding: obvious once read, easy to miss otherwise.

**Do not let research replace measurement.** It tells you what is likely and
what to watch for; it does not tell you what YOUR system is doing. Rules 1-6
below still apply in full — a mechanism that is correct in the literature and
never executes here is still not the cause.

**Cite what you used.** Put the source in the code comment or the FUTURE_WORK
entry, so the next person can check whether the reasoning still holds rather
than re-deriving it. `PREVIOUS_KEY_GRACE` naming WireGuard is the pattern.

## 1. Get a baseline BEFORE you blame

A failure appearing after a change is not evidence the change caused it.

Before attributing a symptom to a change — and *especially* before reverting —
reproduce it with the change absent. If it still fails, the change is not the
cause and you have saved yourself the revert and the false commit message.

> Cost of the check: usually one command. Cost of skipping it on 2026-08-03: a
> revert, a public commit blaming the wrong thing, and a correction commit.

This applies with full force when the change is *yours* and *recent*. That is
when the hypothesis feels strongest and is least tested.

## 2. Absence of evidence is only evidence of absence from a complete source

Before treating "X is not in the log / list / output" as proof X did not happen,
establish that the source would have shown it:

- **Bounded ring buffers** (`activity_history`, the diagnostics recent-activity
  list) drop old entries. A busier peer's events displace a quieter one's.
- **Log level.** Two branches of the pipeline router log at `debug!` while nodes
  run at `info` — "no fallback line in the log" meant nothing at all.
- **Filtered greps.** A pattern that does not match the actual wording, or a
  `tail` that cuts the window, reads identically to a thing not happening
  (gotcha #228 predates this file and is the same error).

If the source is lossy, say so in the writeup rather than reasoning from it.

## 3. A measurement window must be at steady state

This system converges after a restart: the shard registry fills from gossip over
minutes, peers reconnect on backoff, latency samples start empty. A measurement
taken inside that window describes the window, not the system.

Recording "the parallax router never runs" came from counting fallbacks entirely
inside a post-restart period. It runs fine once the registry has filled.

**Before generalising from a count, state the uptime it was taken at**, and
re-take it later if the number matters.

## 4. Verify the mechanism fired, not just that the outcome changed

An outcome can improve for reasons unrelated to the fix.

A dead-peer disconnect test "passed" in 27 seconds — but the new code's warning
was absent, because that connection was QUIC, which libp2p drops natively. The
fix under test had not run. The outcome was right and proved nothing.

**Assert on the mechanism**: the log line the change emits, the branch it takes,
the value it computes. If the change is supposed to fire, prove it fired.

## 5. Check the test fails without the fix

A test that passes before and after is not a regression test, however carefully
worded.

Two of the session's tests were vacuous. One asserted a message still decrypted
after a session downgrade — it passed with the fix removed, because a *different*
fix in the same round decrypted it under the previous key and hid the downgrade.
Rewritten to assert the session KEY, it fails without the fix.

**Toggle the fix off and watch the test go red.** Reverting the whole change with
`git stash` does not count: it removes the test too, and "0 tests ran" is not a
failure.

## 6. One symptom, several contributing factors

Post-hoc attribution to a single root cause is itself a known error mode (Cook,
1998). The 8B routing failure had at least three contributors — shard holdings
that had moved, a peer missing from the candidate set, and an unreliable third
party — and naming any one of them as *the* cause produced a wrong fix.

When a fix does not resolve a symptom, that is information: it means a
contributor remains, not that the fix was pointless.

## 7. When you get it wrong, correct it where the claim lives

Commits and changelog entries are public and relayed to Discord. A wrong causal
claim in one is read by people deciding whether to upgrade.

Push a correction commit naming what was wrong and what is actually true. Do not
silently amend, and do not leave it standing because the code is fine. The
project already carries several such corrections; they are cheap and they are why
the record can be trusted.

## The check, in one line

**Before you claim X caused Y: can you make Y stop by removing X, and come back
by restoring it?** If not, you have a correlation and a story.
