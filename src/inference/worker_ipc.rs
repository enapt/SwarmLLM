//! IPC protocol between the SwarmLLM daemon and model worker subprocesses.
//!
//! Message framing:
//!   [4 bytes LE: json_len][json_len bytes: JSON header]
//!   [4 bytes LE: payload_len][payload_len bytes: raw bytes (0 if none)]

use serde::{Deserialize, Serialize};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use uuid::Uuid;

use crate::types::{
    ModelId, NetworkFinishReason, PrefixBlockEntry, SamplingParams, TensorFormat,
    TensorParallelMeta,
};

use crate::inference::router::TokenLogProbEntry;

// Max framing sizes
const MAX_HEADER: u32 = 64 * 1024 * 1024;
const MAX_PAYLOAD: u32 = 512 * 1024 * 1024;

fn is_zero_u32_vision(v: &u32) -> bool {
    *v == 0
}

/// Message from daemon → worker.
#[derive(Serialize, Deserialize, Debug)]
#[serde(tag = "t")]
pub enum DaemonMsg {
    /// Single-step forward pass (for distributed inference).
    /// Binary payload = activation bytes.
    Forward(IpcForward),
    /// Batched forward pass — multiple requests folded into one IPC call.
    /// Worker processes each request (v1: sequentially; v2: stacked tensor)
    /// and returns one result per request in order. Binary payload is the
    /// concatenation of each request's activation bytes, with `activation_lens`
    /// providing the slice boundaries.
    BatchForward {
        requests: Vec<IpcForward>,
        activation_lens: Vec<u32>,
    },
    /// Full generation loop (for local API inference).
    /// Worker tokenizes, runs prefill + decode, streams tokens back.
    Generate(IpcGenerate),
    /// Guess the next tokens with this (small) model on behalf of a bigger one
    /// — the drafter of speculation across computers (`pipeline::dsd`) run in
    /// this engine from the model's shards. Answered with `WorkerMsg::Drafted`.
    Draft(IpcDraft),
    /// Guess the next tokens of a chain whose first and last segments this
    /// worker holds, through a stand-in for the far layers between them
    /// (`IpcHeadGuess`). Answered with `WorkerMsg::Drafted`.
    HeadGuess(IpcHeadGuess),
    /// Unload a specific layer range (free its GPU memory within the worker).
    Unload {
        layer_start: usize,
        layer_end: usize,
    },
    /// Where a range this worker is about to load goes — sent by the daemon's
    /// growth (`ModelProcessPool::grow_worker`) before the range's first
    /// forward, so a later range is placed by the room each device has NOW
    /// rather than by the spawn's `--gpu-layers` / `--cpu-moe-layers`
    /// (FUTURE_WORK #265). `None` places it as the spawn said. No reply.
    PlaceRange {
        layer_start: usize,
        layer_end: usize,
        placement: Option<crate::inference::split::RangePlacement>,
    },
    /// Read the card's free memory as a load in this worker would read it
    /// (`split::card_free_for_a_load`), before anything is loaded — sent by
    /// the spawn of a card/processor split right after `Ready`, so the daemon
    /// sizes the split by the figure the load's line is drawn on rather than
    /// by its own room (#365). Answered with `WorkerMsg::CardFree`. Sent only
    /// by a daemon to a worker spawned from its own binary or a newer one, so
    /// no worker that cannot read it ever receives it.
    ReadCardFree,
    /// Read a range's layers now, before any forward names them — sent when a
    /// peer's split asks about a prompt this node's range will run
    /// (`ModelProcessPool::load_ahead_for_a_peer`), so the machines of a cold
    /// split load side by side instead of each waiting for the one before it
    /// to finish loading and send its first forward. A worker loads only on a
    /// forward otherwise. No reply: the range's first forward queues behind
    /// the load and finds it done, or meets the load's error itself.
    LoadRange {
        model_id: ModelId,
        layer_start: usize,
        layer_end: usize,
    },
    /// Item 8 Phase 2b: daemon's reply to a `WorkerMsg::PrefixFetchProbe`.
    /// When `present == true`, the BLAKE3-verified `KvSnapshot` bytes are
    /// carried in the IPC binary-payload slot (not inside this JSON header).
    /// `present == false` means miss / timeout / verification failure.
    /// `matched_tokens` is the number of prompt tokens covered by the
    /// snapshot (equal to the chained-hash block boundary the daemon
    /// resolved against).
    PrefixFetchResult {
        request_id: Uuid,
        matched_tokens: u32,
        present: bool,
    },
    /// Item 8 Phase 2b: serving-side request. The daemon received an
    /// inbound `SwarmRequest::PrefixKvFetch` from a peer and needs the
    /// local worker to extract the matching snapshot bytes from its
    /// in-process `PrefixCache`. The worker replies with
    /// `WorkerMsg::PrefixSnapshotResponse`.
    ExportPrefixSnapshot {
        request_id: Uuid,
        model_id: ModelId,
        block_hash: [u8; 32],
    },
    /// Admit a split's prompt over EVERY range this worker will run for the
    /// request, before the pipeline sends its first forward (FUTURE_WORK #263).
    /// Answered with `WorkerMsg::PromptAdmitted` or `WorkerMsg::PromptRefused`.
    ///
    /// Each local segment used to be admitted on its own when its forward
    /// arrived, against the one store the request's earlier segments had
    /// already filled — so the LAST local segment of a long prompt was refused
    /// after the peers between had read it for minutes. Asked once, up front, a
    /// refusal costs no other computer anything. A daemon sends it only to a
    /// worker spawned from its own binary or a newer one, so no worker that
    /// cannot read it ever receives it.
    AdmitPrompt(IpcAdmitPrompt),
    /// Abandon a request the daemon no longer wants a reply for.
    ///
    /// Sent when a `forward` / `generate` future is dropped before it
    /// completes: the client disconnected, or a `tokio::select!` timeout fired.
    /// Without it the worker keeps
    /// computing — for a `Generate` that means generating hundreds of tokens
    /// nobody will read, at full GPU cost.
    ///
    /// Best-effort and idempotent. A cancel for an unknown or already-finished
    /// request is dropped; a cancel that arrives before the request does is
    /// remembered briefly so the request is skipped on arrival.
    CancelRequest { request_id: Uuid },
    /// Release the KV cache this request holds. The request is over.
    ///
    /// The worker's KV is keyed by REQUEST id — `session_id` rides along on
    /// `IpcGenerate` but nothing reads it — so a finished request's entry can
    /// never be read again by anything, whatever carried it. Multi-turn reuse
    /// is the prefix cache's job, and that snapshot is taken before the entry
    /// is cleared.
    ///
    /// The `Generate` handlers already clear their own on the way out. The
    /// forward path had nothing: a segment's KV sat fully charged against the
    /// shared budget until the ten-minute idle sweep, so a handful of finished
    /// turns could refuse the next prompt on a small node — reported live with
    /// `live_mb` identical across three refusals in two different conversations
    /// (report #019).
    ///
    /// Best-effort and idempotent, like `CancelRequest`: an unknown id is a
    /// no-op.
    ReleaseRequestKv { request_id: Uuid },
    /// Graceful shutdown — worker exits cleanly.
    Shutdown,
}

/// Set by the daemon in every worker's environment: this daemon decodes
/// `WorkerMsg::CardAllocationProbe`. See that variant for why a worker must not
/// send it otherwise.
pub(crate) const DAEMON_READS_CARD_PROBE: &str = "SWARMLLM_DAEMON_READS_CARD_PROBE";

/// Whether the daemon that spawned this worker reads the card probe's report.
pub(crate) fn daemon_reads_card_probe(v: Option<&std::ffi::OsStr>) -> bool {
    v.is_some_and(|v| v == "1")
}

/// Set by the daemon in every worker's environment: this daemon decodes
/// `WorkerMsg::ModelLoadTimed`. Gated for the same reason as
/// [`DAEMON_READS_CARD_PROBE`] and read by the same rule
/// ([`daemon_reads_card_probe`]).
pub(crate) const DAEMON_READS_LOAD_TIMING: &str = "SWARMLLM_DAEMON_READS_LOAD_TIMING";

/// Set by the daemon in every worker's environment: this daemon decodes
/// `WorkerMsg::ContextFits`. Gated and read like [`DAEMON_READS_LOAD_TIMING`].
pub(crate) const DAEMON_READS_CONTEXT_FITS: &str = "SWARMLLM_DAEMON_READS_CONTEXT_FITS";

/// Message from worker → daemon.
#[derive(Serialize, Deserialize, Debug)]
#[serde(tag = "t")]
pub enum WorkerMsg {
    /// Worker connected and ready.
    Ready,
    /// Single-step forward result (for Forward requests).
    /// Binary payload = activation bytes (if has_activations).
    LayerResult(IpcLayerResult),
    /// Batched forward results — one per request in `BatchForward.requests`.
    /// `activation_lens[i]` gives the byte length of request i's activations
    /// within the concatenated binary payload (0 if no activations).
    BatchResult {
        results: Vec<IpcLayerResult>,
        activation_lens: Vec<u32>,
    },
    /// The guesses a `DaemonMsg::Draft` asked for, in order. For a tree
    /// (`IpcDraft::tree`), each guess's parent among them in the wire form of
    /// `LayerForward::draft_parents` (0: it follows the last token read);
    /// empty for a chain.
    Drafted {
        request_id: Uuid,
        tokens: Vec<u32>,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        parents: Vec<u16>,
        /// For a tree, each guess's path log-probability under the drafter
        /// (`split::draft::GrownTree::path_logp`); empty for a chain.
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        scores: Vec<f32>,
        /// One per guess where `IpcDraft::runner_ups` asked; else empty.
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        runner_ups: Vec<RunnerUp>,
    },
    /// A `DaemonMsg::AdmitPrompt` fitted: the request's cache is promised over
    /// every range it named, and their forwards ask nothing more. Sent only in
    /// answer to that message, so only to a daemon that reads it.
    PromptAdmitted { request_id: Uuid },
    /// A `DaemonMsg::AdmitPrompt` did not fit. `message` is the refusal the
    /// owner reads (`LocalMemoryUnavailable`'s); `layers_that_fit` is how many
    /// of this model's layers' caches the room holds at this prompt's length —
    /// the fact the re-plan needs and a bare refusal never gave it (#263).
    PromptRefused {
        request_id: Uuid,
        message: String,
        layers_that_fit: u32,
    },
    /// A single decoded token (for streaming Generate).
    Token {
        request_id: Uuid,
        token_id: u32,
        text: String,
        is_eos: bool,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        logprob: Option<f32>,
    },
    /// Generation complete (follows the last Token message).
    GenerateDone {
        request_id: Uuid,
        prompt_tokens: usize,
        completion_tokens: usize,
        finish_reason: String,
        /// The user-provided stop sequence that matched, if any. Populated
        /// only when `finish_reason == "stop"` and the match came from
        /// `SamplingParams.stop` (not EOS). Carried back to the API layer
        /// so Anthropic clients see the actual matched sequence rather
        /// than `null`. Optional + default-skipping keeps wire-compat with
        /// older worker builds.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        matched_stop_sequence: Option<String>,
    },
    /// How long the worker's fresh card allocations took when it first picked
    /// the card (`cuda_pool::probe_once`), sent once, after the message that
    /// loaded a model. Side-band like `Progress`: the daemon's
    /// `cuda_pool::SlowCardNotice` tells the owner when it is slow (#762).
    ///
    /// ⚠ **Sent only when the daemon said it reads it** ([`DAEMON_READS_CARD_PROBE`]).
    /// A worker is spawned from the binary on disk NOW (`current_exe_path`,
    /// gotcha #188), which after an update that has not restarted the daemon is
    /// NEWER than the daemon; a daemon before v0.3.225 cannot decode this
    /// variant and fails on the JSON before reading the payload, so the stream
    /// is lost and the worker evicted on every load (pre-release review,
    /// 2026-10-01). From v0.3.225 the reader skips a message it does not know
    /// (`recv_worker`, FUTURE_WORK #150) — but a daemon that old still runs in
    /// the field after an update it has not restarted into, so gate every new
    /// worker-to-daemon message the same way while any can: the IPC twin of
    /// gating a new wire message on a peer's `features`.
    CardAllocationProbe { took_ms: u32 },
    /// How long one model load took this worker and how much of the model it
    /// read (`bytes`: the model's size × the share of its layers loaded — the
    /// basis the planner prices a load on). Timed inside the load itself, so a
    /// process start or a card's first context is not in it (FUTURE_WORK #212's
    /// reverted attempt recorded those). Side-band, sent after the message that
    /// loaded the model; the daemon folds it into the rate it advertises
    /// (`NodeCapability::model_load_ms_per_gib`, FUTURE_WORK #129).
    ///
    /// ⚠ Sent only when the daemon said it reads it
    /// ([`DAEMON_READS_LOAD_TIMING`]) — see `CardAllocationProbe`.
    ModelLoadTimed { bytes: u64, ms: u32 },
    /// The longest conversation, prompt and reply together, the model this
    /// worker holds WHOLE admits with nothing else held
    /// (`model_worker::whole_model_context_fits`); `None` while it holds no
    /// whole model, or no budget is known for it. Side-band, sent when it
    /// changes — read after a message that loaded or dropped a model, and at
    /// most once a minute after (`model_worker::CONTEXT_FITS_EVERY`). The daemon
    /// keeps it on the worker's handle, and `/v1/models` reports it in place of
    /// a longer declared context (FUTURE_WORK #252 (1)).
    ///
    /// ⚠ Sent only when the daemon said it reads it
    /// ([`DAEMON_READS_CONTEXT_FITS`]) — see `CardAllocationProbe`.
    ContextFits { tokens: Option<u32> },
    /// The answer to `DaemonMsg::ReadCardFree`: the card's free memory as a
    /// load here reads it, in bytes; `None` without a card or a reading.
    /// Sent only in answer to that message, so only to a daemon that reads it.
    CardFree { free_bytes: Option<u64> },
    /// Error for a specific request.
    ///
    /// `fatal` marks a failure that corrupted or exhausted the worker's device
    /// state rather than just this one request — a CUDA OOM, a driver error, an
    /// illegal memory access. The daemon evicts (and thereby kills) the worker
    /// on a fatal error so the OS reclaims its VRAM; a non-fatal error leaves
    /// the loaded model resident for the next request. Defaults to `false` for
    /// wire-compat with workers that predate the field.
    Error {
        request_id: Uuid,
        message: String,
        #[serde(default)]
        fatal: bool,
        /// This node's own memory budget refused the work, rather than anything
        /// going wrong.
        ///
        /// Carried as a flag because the TYPE does not survive this hop —
        /// everything arrives as a string — and because the daemon must not
        /// re-derive it from the wording: `LocalMemoryUnavailable` deliberately
        /// shares `ServiceUnavailable`'s message shape, so that a peer refusing
        /// for memory is still treated as a peer refusing. The distinction is
        /// local, and this is the one place that can state it without guessing:
        /// the worker that refused is ours.
        ///
        /// It is the one local failure the router re-plans, because it is the
        /// one where the re-plan is handed a fact it did not have. Without it
        /// the refusal read as an ordinary local failure and the request died
        /// with a priced multi-peer alternative sitting unused (report #019).
        #[serde(default)]
        local_memory_refusal: bool,
    },
    /// Item 8 Phase 1: notify the daemon that the worker just inserted (or
    /// refreshed) prefix-cache entries for `model_id`. The daemon broadcasts
    /// this as a `SwarmMessage::PrefixCacheAnnounce` and updates its own
    /// cross-node index. Carries no `request_id` and is fanned out by the
    /// reader actor through a dedicated channel rather than per-request
    /// response routing.
    PrefixManifestUpdate {
        model_id: ModelId,
        blocks: Vec<PrefixBlockEntry>,
    },
    /// Item 8 Phase 2b: worker-initiated probe for a cross-node prefix KV
    /// fetch. Carries the chained-hash manifest of the current prompt's
    /// leading blocks; the daemon picks the longest-prefix peer match,
    /// fetches + verifies + returns bytes via `DaemonMsg::PrefixFetchResult`.
    /// `request_id` correlates the probe with the result.
    PrefixFetchProbe {
        request_id: Uuid,
        model_id: ModelId,
        blocks: Vec<PrefixBlockEntry>,
        /// The probing request's caller asked to keep it on this computer
        /// (#251 (2b)), so the daemon asks no peer: the probe's scope is
        /// `pool::scope::request_scope` under that instruction.
        #[serde(default)]
        keep_here: bool,
    },
    /// Item 8 Phase 2b: worker's reply to `DaemonMsg::ExportPrefixSnapshot`.
    /// When `present == true`, the serialized `KvSnapshot` bytes are carried
    /// in the IPC binary-payload slot (not inside this JSON header).
    /// Snapshots can be tens of MB; JSON-encoding a `Vec<u8>` bloats by ~5×
    /// and overflows the 64 MiB header cap. `present == false` means the
    /// worker couldn't produce a snapshot (eviction race / miss).
    PrefixSnapshotResponse { request_id: Uuid, present: bool },
    /// Progress on a request that has not produced its first token yet.
    ///
    /// Prefill is linear in prompt length and is ~99% of a long request, so a
    /// large prompt on a modest machine can run for minutes before anything
    /// reaches the client — indistinguishable, from outside, from a hang. This
    /// carries enough to say what the worker is doing and how far along it is;
    /// the daemon derives an ETA from the observed rate (it has the timing and
    /// the trace, the worker has neither).
    ///
    /// Additive and fire-and-forget: a worker that never sends it (an older
    /// binary) simply leaves the request's phase unset. (A daemon that does not
    /// know a message skips it only from v0.3.225 — see `CardAllocationProbe`.)
    Progress {
        request_id: Uuid,
        phase: ProgressPhase,
        /// Prompt tokens consumed so far.
        done: u32,
        /// Prompt tokens in total. Zero when not meaningful for the phase.
        total: u32,
    },
    /// Worker is about to exit.
    Bye,
}

/// What a not-yet-streaming request is currently doing.
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ProgressPhase {
    /// Reading model weights off disk and onto the device. Has no token
    /// counter — the cost is per byte, not per token.
    LoadingModel,
    /// Reading the prompt. `done`/`total` are prompt tokens.
    Prefill,
}

impl ProgressPhase {
    /// Short label for logs, API payloads and the dashboard. Stable — the
    /// frontend and the admin API both key off these strings.
    pub fn as_str(self) -> &'static str {
        match self {
            ProgressPhase::LoadingModel => "loading_model",
            ProgressPhase::Prefill => "prefill",
        }
    }
}

/// `DaemonMsg::AdmitPrompt`'s body: one request's prompt, and every layer range
/// this worker will run for it in the plan about to start.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct IpcAdmitPrompt {
    pub request_id: Uuid,
    pub model_id: ModelId,
    pub layer_ranges: Vec<(u32, u32)>,
    /// The whole prompt, in positions — what every range's cache will hold
    /// once the prompt pass is done, a kept opening (#10) included.
    pub prompt_tokens: u32,
    /// Weigh the prompt and record NOTHING (`model_worker::Admitting::OnlyWeigh`):
    /// a PEER's question (`PromptAdmissionAsk`, FUTURE_WORK #252 (b)). No room
    /// is held for a machine that may never send its forwards, no conversation
    /// is ended and no saved prompt evicted to answer it, and `request_id` is a
    /// fresh correlation id, never the request's own — an ask naming a request
    /// that runs here must not displace its forwards' answers.
    #[serde(default)]
    pub only_weigh: bool,
}

/// Forward-pass request header (activation bytes are the binary payload).
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct IpcForward {
    pub request_id: Uuid,
    pub sequence_num: u32,
    pub index_pos: u32,
    pub format: TensorFormat,
    pub model_id: ModelId,
    pub layer_range: (u32, u32),
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tp_meta: Option<TensorParallelMeta>,
    /// Byte length of the vision-embedding prefix at the head of the IPC
    /// binary payload (zero when absent). Vision embeddings used to live
    /// inside this JSON header as `Option<Vec<u8>>`, but serde_json encodes
    /// `Vec<u8>` as a JSON array of integers (~5× bloat). LLaVA-class
    /// mmproj output can exceed 1 MiB before zstd compression and tens of
    /// MiB after decompression in pathological cases, and the same latent
    /// bomb as spec_logits (gotcha #24) can push the JSON header past
    /// `MAX_HEADER` (64 MiB). Payload layout: `[vision_bytes][activation_bytes]`
    /// with the vision prefix length given here.
    #[serde(default, skip_serializing_if = "is_zero_u32_vision")]
    pub vision_embeddings_len: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub requester_node_id: Option<[u8; 32]>,
    #[serde(default)]
    pub pre_embedded: bool,
    pub sampling: SamplingParams,
    /// LoRA adapter ID to apply during inference. The worker loads the adapter
    /// from the data_dir/adapters/ directory on first use.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub adapter_id: Option<String>,
    /// Speculative decoding: γ draft tokens. When non-empty, the worker runs
    /// a multi-position forward and returns per-position logits.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub draft_tokens: Vec<u32>,
    /// OpenAI-style frequency_penalty / presence_penalty history: the
    /// completion-so-far token IDs. Populated by the daemon coordinator on
    /// the final-segment forward when `sampling.frequency_penalty != 0` or
    /// `sampling.presence_penalty != 0`; empty otherwise (default no-op,
    /// no wire bloat). Worker passes this to
    /// `apply_repetition_penalties` before sampling.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub generated_ids: Vec<u32>,
    /// Coordinator wants per-position logit vectors populated on the result.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub spec_logits_requested: bool,
    /// Walk `draft_tokens` against those logits here and answer with token ids
    /// (`LayerForward::spec_walk_at_tail`).
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub spec_walk_at_tail: bool,
    /// The request's shared noise seed for the walk (`LayerForward::coupling_seed`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub coupling_seed: Option<u64>,
    /// `LayerForward::stream_seq`: this forward is one of a stream of verifies
    /// that arrive back to back, so the worker checks that the cache holds
    /// exactly the positions before `index_pos` (after any truncation) and
    /// refuses the forward otherwise — a chunk of a stream run out of order
    /// would be written at the cache's length, not its own position.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stream_seq: Option<u32>,
    /// KV cache truncation: if `Some(L)`, worker truncates per-request KV to
    /// L sequence positions BEFORE running the forward. Used by speculative
    /// partial-accept fixup.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub truncate_kv_to: Option<u32>,
    /// A split's prompt pass that lets this segment keep its part of the prompt
    /// between turns (`LayerForward::prompt_cache`, FUTURE_WORK #10).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_cache: Option<crate::types::PromptCacheHint>,
    /// This forward is one piece of a prompt pass read in pieces
    /// (`LayerForward::prompt_span`, FUTURE_WORK #171).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_span: Option<crate::types::PromptSpan>,
    /// The prompt pass's pictures whose rows attend both ways
    /// (`LayerForward::image_spans`, #115 (2b)).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub image_spans: Vec<crate::types::PromptSpan>,
    /// The shape of a tree of guesses (`LayerForward::draft_parents`, #134).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub draft_parents: Vec<u16>,
    /// Where a tree that flows starts (`LayerForward::draft_tree_from`, #134
    /// (b)): `draft_parents` is then the whole tree from there, and the last
    /// segment answers one draw per row.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub draft_tree_from: Option<u32>,
    /// This forward belongs to a request THIS machine's owner sent — the
    /// segment of it this node runs inside a pipeline it coordinates — so its
    /// prompt pass may read on the owner's width (`process_pool::Requester`).
    /// The `Generate` path carries the same fact on `IpcGenerate::for_the_owner`;
    /// without it here the owner's prompt fell back to the contribution width
    /// whenever the request was split. Set only by the daemon from its own
    /// call site, never read off the network: a peer's segment has no way to
    /// set it.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub for_the_owner: bool,
    /// The far run `[a, b)` of a chain whose first and last segments this
    /// worker both holds, which its coordinator guesses through
    /// (`DaemonMsg::HeadGuess`): this forward's hidden state at the run's
    /// near end (a first segment's output) or far end (a last segment's input)
    /// is kept to learn the run's stand-in (`model_worker::FarRuns`). Set only
    /// by the daemon, for a request it coordinates
    /// (`ModelProcessPool::learn_far_run`), never read off the network.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub far_run: Option<(u32, u32)>,
}

/// A guess at the head of a chain whose two ends this worker holds
/// (`DaemonMsg::HeadGuess`): commit what the request's forwards kept below
/// `index_pos`, cut both ends' caches back to it, then guess `gamma` tokens
/// after `token` — each through the first segment, the far run's stand-in
/// and the last segment — and cut both back to `index_pos` again. Answered
/// with `WorkerMsg::Drafted` (no guesses where the stand-in has not learned
/// enough yet).
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct IpcHeadGuess {
    pub request_id: Uuid,
    pub model_id: ModelId,
    pub first_range: (u32, u32),
    pub last_range: (u32, u32),
    pub index_pos: u32,
    pub token: u32,
    pub gamma: u32,
}

/// Forward-pass result header.
///
/// The binary payload slot carries ONE of (mutually exclusive):
/// - activation bytes (when `has_activations`)
/// - raw f32 speculative logits, row-major `spec_logits_dims.0 × .1`
///   (when `has_spec_logits`)
/// - nothing (when both flags are false, e.g. last-segment token emit)
///
/// Speculative logits used to live inside this JSON header as
/// `Vec<Vec<f32>>`, but serde_json encodes each f32 as ~7 bytes of
/// decimal text — a 5–7× bloat over binary. On large-vocab models
/// (Qwen2.5 at 151 K, γ=4) that approaches `MAX_HEADER` (64 MiB) and
/// on γ=8 blows past it. Same class of bug as gotcha #24
/// (`Vec<u8>` in JSON header).
#[derive(Serialize, Deserialize, Debug)]
pub struct IpcLayerResult {
    pub request_id: Uuid,
    pub token_ids: Vec<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<NetworkFinishReason>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub format: Option<TensorFormat>,
    #[serde(default)]
    pub sealed: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sealed_payload: Option<Vec<u8>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<Vec<TokenLogProbEntry>>,
    /// User-provided stop sequence that triggered termination, if any.
    /// Populated by the worker when `finish_reason == Stop` and the match
    /// came from `SamplingParams.stop`. Plumbed up to `LayerResult` so the
    /// distributed-path coordinator can carry it back to API clients.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub matched_stop_sequence: Option<String>,
    /// True if binary payload contains output activation bytes.
    #[serde(default)]
    pub has_activations: bool,
    /// True if binary payload contains raw f32 spec-verify logits.
    /// Mutually exclusive with `has_activations`.
    #[serde(default)]
    pub has_spec_logits: bool,
    /// `(n_positions, vocab_size)` — only meaningful when
    /// `has_spec_logits` is true. Used to reshape the f32 payload back
    /// into `Vec<Vec<f32>>` on the receiving side.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub spec_logits_dims: Option<(u32, u32)>,
    /// The prompt blocks this segment stored for the next turn
    /// (`LayerResult::prompt_blocks_stored`). The ONE result field the worker
    /// produces that the daemon carries on: the rest of `LayerResult`'s
    /// network-only fields are stamped by the daemon.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_blocks_stored: Option<u32>,
}

/// Encode `Vec<Vec<f32>>` as a flat little-endian f32 byte buffer,
/// row-major. Companion to `decode_spec_logits` — sender side.
pub fn encode_spec_logits(logits: &[Vec<f32>]) -> (Vec<u8>, (u32, u32)) {
    let n_positions = logits.len() as u32;
    let vocab_size = logits.first().map(|r| r.len()).unwrap_or(0) as u32;
    let mut bytes = Vec::with_capacity((n_positions as usize) * (vocab_size as usize) * 4);
    for row in logits {
        for v in row {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
    }
    (bytes, (n_positions, vocab_size))
}

/// Decode a flat little-endian f32 payload back into `Vec<Vec<f32>>`
/// given the `(n_positions, vocab_size)` dims recorded in the header.
/// Returns an error when the payload length disagrees with dims.
pub fn decode_spec_logits(bytes: &[u8], dims: (u32, u32)) -> Result<Vec<Vec<f32>>, String> {
    let n = dims.0 as usize;
    let v = dims.1 as usize;
    let expected = n
        .checked_mul(v)
        .and_then(|x| x.checked_mul(4))
        .ok_or_else(|| "spec_logits dims overflow usize".to_string())?;
    if bytes.len() != expected {
        return Err(format!(
            "spec_logits payload len {} mismatch — expected {} (n={}, v={})",
            bytes.len(),
            expected,
            n,
            v
        ));
    }
    let mut out = Vec::with_capacity(n);
    for row_bytes in bytes.chunks_exact(v * 4) {
        let mut row = Vec::with_capacity(v);
        for chunk in row_bytes.as_chunks::<4>().0.iter() {
            row.push(f32::from_le_bytes(*chunk));
        }
        out.push(row);
    }
    Ok(out)
}

/// Generate request (full decode loop in the worker).
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct IpcGenerate {
    pub request_id: Uuid,
    pub model_id: ModelId,
    pub layer_range: (u32, u32),
    /// Already-formatted prompt (chat template applied by caller).
    pub prompt: String,
    pub sampling: SamplingParams,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
    /// This machine's owner made the request (`process_pool::Requester`): its
    /// prompt is read on every physical core rather than within the
    /// contribution level. The worker records it on the request's KV
    /// bookkeeping (`KvCacheStore::mark_owner_request`).
    #[serde(default)]
    pub for_the_owner: bool,
    /// The request's caller asked to keep it on this computer (#251 (2b)):
    /// its prompt's cache is saved for reuse HERE only — never announced to a
    /// peer, exported to one, or looked for on one (`PrefixCache`'s
    /// `kept_here`, `WorkerMsg::PrefixFetchProbe::keep_here`). Both ends are
    /// one build: a worker only ever talks to the daemon that spawned it (its
    /// socket is named afresh at each spawn, and the daemon's end closes when
    /// an update `exec`s the new binary), so an older worker can never drop
    /// the field for a newer daemon.
    #[serde(default)]
    pub keep_here: bool,
    /// The JSON the reply is held to (#336) — `sampling.reply_grammar`, which
    /// is skipped wherever `SamplingParams` is serialized (a peer is not told
    /// it), carried across this hop beside it. Built only by
    /// [`IpcGenerate::new`] and put back only by
    /// [`IpcGenerate::rejoin_grammar`], so neither end can drop it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub grammar: Option<crate::types::ReplyGrammar>,
}

impl IpcGenerate {
    /// The one way the daemon builds a generate request: the reply grammar
    /// rides beside the sampling it cannot cross inside.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        request_id: Uuid,
        model_id: ModelId,
        layer_range: (u32, u32),
        prompt: String,
        sampling: SamplingParams,
        session_id: Option<String>,
        for_the_owner: bool,
        keep_here: bool,
    ) -> Self {
        let grammar = sampling.reply_grammar.as_deref().cloned();
        Self {
            request_id,
            model_id,
            layer_range,
            prompt,
            sampling,
            session_id,
            for_the_owner,
            keep_here,
            grammar,
        }
    }

    /// Put the reply grammar back into the sampling, as the daemon held it —
    /// the worker's first act on a generate request, before either path
    /// (the batched table or `handle_generate`) reads the sampling.
    pub fn rejoin_grammar(&mut self) {
        if let Some(grammar) = self.grammar.take() {
            self.sampling.reply_grammar = Some(std::sync::Arc::new(grammar));
        }
    }
}

/// A drafting call (`DaemonMsg::Draft`).
///
/// The worker keeps the request's cache between calls, so each call carries
/// only what changed: cut the cache back to `keep` positions — what the last
/// round's check confirmed of the guesses this model made — read `append` (the
/// tokens since, ending with the one the next guess follows), then guess
/// `gamma` tokens, reading each back in so the next can follow it. After the
/// call the cache holds `keep + append.len() + gamma - 1` positions; the caller
/// tracks how many of them the next check confirms.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct IpcDraft {
    /// The speculating ATTEMPT's key (`pipeline::engine_drafter::draft_key`),
    /// not the target request's id — a router retry reuses that id, and two
    /// attempts sharing one drafter cache clobbered each other.
    pub request_id: Uuid,
    pub model_id: ModelId,
    pub layer_range: (u32, u32),
    pub keep: u32,
    pub append: Vec<u32>,
    pub gamma: u32,
    /// The target request's sampling parameters, for a guess drawn with the
    /// shared noise; unused for a greedy guess.
    pub sampling: SamplingParams,
    /// The reply so far, which the sampler's penalties read.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub history: Vec<u32>,
    /// The request's shared noise seed (`inference::coupled_noise`): guess the
    /// token the target's sampler will DRAW rather than its most likely one.
    /// `None` guesses the argmax.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub coupling_seed: Option<u64>,
    /// See `IpcForward::for_the_owner`.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub for_the_owner: bool,
    /// Stop guessing after a guess this model gives less than this probability
    /// (`SplitModel::draft_after`). `None` guesses all `gamma`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stop_below: Option<f32>,
    /// Grow a TREE of guesses instead of a chain (`SplitModel::draft_tree_after`,
    /// #134); `gamma` and `stop_below` are unused then. Afterwards the cache
    /// holds `keep + append.len()` positions — no guess; while it grows, the
    /// seed's rows and the ones added.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tree: Option<TreeAsk>,
    /// Grow a tree that FLOWS (#134 (b), `SplitModel::grow_flow`): kept in the
    /// worker between calls for this attempt, changed by the requester's ops,
    /// answered with proposed rows (`Drafted::parents` = parent row + 1).
    /// `gamma`, `stop_below`, `tree` and `history` are unused then.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub flow: Option<FlowAsk>,
    /// Answer each chain guess's runner-up too (`Drafted::runner_ups`), for a
    /// stream's spare restart (#381). Greedy guesses only: unused with a
    /// `coupling_seed`, a `tree` or a `flow`.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub runner_ups: bool,
}

/// The drafter's view of one chain guess's position (`IpcDraft::runner_ups`):
/// how sure it was of its guess, and the token it would have guessed instead
/// — what a stream's spare restart (#381, `dsd_stream`) bets the check draws
/// there if it refuses the guess.
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq)]
pub struct RunnerUp {
    /// The drafter's probability for its own guess.
    pub guess_p: f32,
    /// Its second choice there.
    pub token: u32,
    /// Its probability for `token`.
    pub p: f32,
}

/// A drafting call of a tree that flows (`IpcDraft::flow`,
/// `split::draft::FlowStep`).
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Default)]
pub struct FlowAsk {
    /// Start the tree afresh from this token — row 0, at `keep + append.len()`
    /// once `append` (the context since `keep`, not this token) is read.
    /// `None` continues this attempt's tree; `keep` and `append` are unused.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub start: Option<u32>,
    /// The requester's changes since the last call, in its order. Never a
    /// proposal's push: a proposal is a row of both trees once made.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub ops: Vec<crate::inference::draft_tree::FlowOp>,
    /// Rows to propose a level, at most.
    pub rows: u32,
    /// Levels to grow, each read by the next (`FlowStep::levels`).
    #[serde(default)]
    pub levels: u32,
    /// Propose none past this many live rows.
    pub live_most: u32,
}

/// The tree of guesses a drafting call grows (`IpcDraft::tree`,
/// `split::draft::TreeGrowth`).
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Default)]
pub struct TreeAsk {
    /// Guesses the call adds to `seed`.
    pub rows: u32,
    /// Guesses read in one forward of the drafter while it grows.
    pub per_step: u32,
    /// Stop adding guesses once the call has run this long: a tree grown while
    /// a check is on its way is wanted when the check's answer is. `None` adds
    /// all `rows`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub grow_for_ms: Option<u32>,
    /// The tree grown so far after the same token, to grow on (empty: grow
    /// from nothing) — in `Drafted`'s form.
    #[serde(default, skip_serializing_if = "TreeSeed::is_empty")]
    pub seed: TreeSeed,
}

/// A tree of guesses handed to a drafting call to grow on (`TreeAsk::seed`):
/// each guess's token, its parent in `LayerForward::draft_parents`' wire form
/// (0: it follows the last token read; `k`: row `k - 1`) and its path's
/// log-probability.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Default)]
pub struct TreeSeed {
    pub tokens: Vec<u32>,
    pub parents: Vec<u16>,
    pub scores: Vec<f32>,
}

impl TreeSeed {
    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty() && self.parents.is_empty() && self.scores.is_empty()
    }

    pub(crate) fn of(tree: &crate::inference::draft_tree::GrownTree) -> Self {
        Self {
            tokens: tree.tokens.clone(),
            parents: tree
                .parents
                .iter()
                .map(|p| p.map_or(0, |p| p + 1))
                .collect(),
            scores: tree.path_logp.clone(),
        }
    }

    pub(crate) fn grown(&self) -> crate::inference::draft_tree::GrownTree {
        crate::inference::draft_tree::GrownTree {
            tokens: self.tokens.clone(),
            parents: self.parents.iter().map(|&w| w.checked_sub(1)).collect(),
            path_logp: self.scores.clone(),
        }
    }
}

/// What a drafting call answered (`WorkerMsg::Drafted`): the guesses, and for
/// a tree each one's parent in `LayerForward::draft_parents`' wire form and
/// its path's log-probability. `parents` and `scores` are empty for a chain.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Drafted {
    pub tokens: Vec<u32>,
    pub parents: Vec<u16>,
    pub scores: Vec<f32>,
    /// Each guess's runner-up, where the call asked (`IpcDraft::runner_ups`).
    pub runner_ups: Vec<RunnerUp>,
}

impl IpcDraft {
    /// The cache this call's reply needs, in positions: what the cache keeps,
    /// what the call appends and the reply's reserve
    /// (`model_worker::drafter_cache_positions`) — what a drafter loaded for
    /// it is charged (`process_pool::Tenancy::Guest`, #275).
    pub fn conversation_positions(&self) -> u64 {
        // A tree that flows keeps its rows in the cache between calls.
        let tree_rows = if self.flow.is_some() {
            crate::inference::draft_tree::MAX_TREE_ROWS
        } else {
            0
        };
        crate::inference::model_worker::drafter_cache_positions(
            (self.keep as usize)
                .saturating_add(self.append.len())
                .saturating_add(tree_rows),
            Some(self.sampling.max_tokens),
        )
    }
}

/// Send a DaemonMsg to the socket. Payload = raw bytes (e.g., activations).
pub async fn send_daemon<W: AsyncWriteExt + Unpin>(
    w: &mut W,
    msg: &DaemonMsg,
    payload: &[u8],
) -> std::io::Result<()> {
    send_framed(w, msg, payload).await
}

/// Send a WorkerMsg to the socket.
pub async fn send_worker<W: AsyncWriteExt + Unpin>(
    w: &mut W,
    msg: &WorkerMsg,
    payload: &[u8],
) -> std::io::Result<()> {
    send_framed(w, msg, payload).await
}

/// Read a DaemonMsg from the socket.
pub async fn recv_daemon<R: AsyncReadExt + Unpin>(
    r: &mut R,
) -> std::io::Result<(DaemonMsg, Vec<u8>)> {
    recv_framed(r).await
}

/// Read a WorkerMsg from the socket.
///
/// A message whose tag this build does not know comes back as an error that
/// [`unknown_worker_msg`] recognises, with its whole frame consumed — so the
/// stream stays aligned and the reader can skip it (FUTURE_WORK #150). A worker
/// is spawned from the binary on disk (gotcha #188), so after an update this
/// daemon has not restarted into it talks to a NEWER worker; one new side-band
/// message used to desynchronise the stream and evict the worker on every load.
/// A message with a KNOWN tag that fails to decode is still fatal: that is a
/// broken stream, not a newer one.
pub async fn recv_worker<R: AsyncReadExt + Unpin>(
    r: &mut R,
) -> std::io::Result<(WorkerMsg, Vec<u8>)> {
    let (header, payload) = recv_frame(r).await?;
    match serde_json::from_slice::<WorkerMsg>(&header) {
        Ok(msg) => Ok((msg, payload)),
        Err(e) => Err(match unknown_tag(&header) {
            Some(tag) => std::io::Error::other(UnknownWorkerMsg { tag }),
            None => std::io::Error::new(std::io::ErrorKind::InvalidData, e),
        }),
    }
}

/// A worker message whose tag this build does not know — from a newer worker.
#[derive(Debug)]
pub struct UnknownWorkerMsg {
    pub tag: String,
}

impl std::fmt::Display for UnknownWorkerMsg {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "a worker message this daemon does not know ({})",
            self.tag
        )
    }
}

impl std::error::Error for UnknownWorkerMsg {}

/// The tag of a message [`recv_worker`] skipped as unknown, if `e` is one.
pub fn unknown_worker_msg(e: &std::io::Error) -> Option<&str> {
    e.get_ref()?
        .downcast_ref::<UnknownWorkerMsg>()
        .map(|u| u.tag.as_str())
}

/// Every [`WorkerMsg`] tag, as its own type: whether a tag is one this build
/// knows is decided by deserialising it, never by reading serde's error text
/// (#295). Kept complete by `every_worker_message_has_a_tag`, whose exhaustive
/// match fails to compile when a `WorkerMsg` variant is added without one.
#[derive(Deserialize, Debug, PartialEq, Eq)]
enum WorkerMsgTag {
    Ready,
    LayerResult,
    BatchResult,
    Drafted,
    PromptAdmitted,
    PromptRefused,
    Token,
    GenerateDone,
    CardAllocationProbe,
    ModelLoadTimed,
    ContextFits,
    CardFree,
    Error,
    PrefixManifestUpdate,
    PrefixFetchProbe,
    PrefixSnapshotResponse,
    Progress,
    Bye,
}

/// The `t` of a header that is a tagged object whose tag no [`WorkerMsgTag`]
/// names; `None` for a known tag or a header with no readable tag.
fn unknown_tag(header: &[u8]) -> Option<String> {
    let v: serde_json::Value = serde_json::from_slice(header).ok()?;
    let tag = v.get("t")?.as_str()?;
    serde_json::from_value::<WorkerMsgTag>(serde_json::Value::String(tag.to_string()))
        .is_err()
        .then(|| tag.to_string())
}

async fn send_framed<W: AsyncWriteExt + Unpin, T: Serialize>(
    w: &mut W,
    msg: &T,
    payload: &[u8],
) -> std::io::Result<()> {
    let json = serde_json::to_vec(msg)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    // SEC: Validate lengths fit in u32 before casting to prevent silent truncation
    let json_len = u32::try_from(json.len()).map_err(|_| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "IPC header too large for u32",
        )
    })?;
    let payload_len = u32::try_from(payload.len()).map_err(|_| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "IPC payload too large for u32",
        )
    })?;
    w.write_all(&json_len.to_le_bytes()).await?;
    w.write_all(&json).await?;
    w.write_all(&payload_len.to_le_bytes()).await?;
    if !payload.is_empty() {
        w.write_all(payload).await?;
    }
    w.flush().await
}

async fn recv_framed<R: AsyncReadExt + Unpin, T: for<'de> Deserialize<'de>>(
    r: &mut R,
) -> std::io::Result<(T, Vec<u8>)> {
    let (header, payload) = recv_frame(r).await?;
    let msg: T = serde_json::from_slice(&header)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    Ok((msg, payload))
}

/// One whole frame — header and payload — read before either is decoded, so a
/// header that does not decode leaves the stream at the next frame.
async fn recv_frame<R: AsyncReadExt + Unpin>(r: &mut R) -> std::io::Result<(Vec<u8>, Vec<u8>)> {
    let mut buf4 = [0u8; 4];
    r.read_exact(&mut buf4).await?;
    let json_len = u32::from_le_bytes(buf4);
    if json_len == 0 || json_len > MAX_HEADER {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("IPC header len {json_len} invalid (must be 1..{MAX_HEADER})"),
        ));
    }
    let mut json_buf = vec![0u8; json_len as usize];
    r.read_exact(&mut json_buf).await?;
    r.read_exact(&mut buf4).await?;
    let payload_len = u32::from_le_bytes(buf4);
    if payload_len > MAX_PAYLOAD {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("IPC payload {payload_len} > max {MAX_PAYLOAD}"),
        ));
    }
    let mut payload = vec![0u8; payload_len as usize];
    if payload_len > 0 {
        r.read_exact(&mut payload).await?;
    }
    Ok((json_buf, payload))
}

/// Does this worker-side error message indicate the worker's device state is
/// no longer trustworthy, rather than just this one request failing?
///
/// This is the single source of truth for the classification — the worker uses
/// it to stamp `WorkerMsg::Error.fatal`, and the daemon re-checks the message
/// text so a worker built before the `fatal` field still gets recycled.
///
/// A `true` verdict means "kill and respawn the worker". Getting it wrong in
/// the `true` direction costs one model reload; getting it wrong in the `false`
/// direction strands the worker's entire VRAM allocation for the lifetime of
/// the daemon, and every retry then competes against memory the leaked worker
/// is still holding. The asymmetry is why the patterns below lean inclusive.
///
/// Deliberately NOT fatal: bad prompts, tokenizer failures, unsupported
/// sampling params, shape mismatches from a malformed request. Those are
/// per-request and the loaded model is still perfectly good.
pub fn worker_error_is_fatal(message: &str) -> bool {
    // Patterns are matched case-insensitively against the full error chain,
    // which for candle looks like:
    //   "Forward: Cuda(DriverError(CUDA_ERROR_OUT_OF_MEMORY, "out of memory"))"
    const FATAL_PATTERNS: &[&str] = &[
        "out of memory",
        "outofmemory",
        "out_of_memory",
        "cuda_error",
        "cuda error",
        "drivererror",
        "illegal memory access",
        "misaligned address",
        "device-side assert",
        "no cuda-capable device",
        "cublas",
        "cudnn",
        "not enough memory",
        "failed to allocate",
        "allocation failed",
        // A GPU placement with no room left for the conversation cache. Fatal
        // in the sense this function means: the worker must be recycled, so the
        // pool can bring the model back on the processor instead of leaving a
        // worker up that refuses every request.
        NO_KV_CACHE_ROOM_MARKER,
    ];
    let lower = message.to_ascii_lowercase();
    FATAL_PATTERNS.iter().any(|p| lower.contains(p))
}

/// Why a GPU failure will repeat verbatim if the same model is retried on the
/// same device — i.e. why the model should be moved to the CPU instead.
///
/// Distinct from [`worker_error_is_fatal`], which asks only whether the WORKER
/// must be recycled. A transient fatal error (an illegal access from a race)
/// deserves a respawn on the GPU; these two do not, because nothing about the
/// retry differs from the attempt that just failed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PermanentGpuFailure {
    /// The allocation does not fit, and the retry would request the same bytes.
    OutOfMemory,
    /// The card is older than the architecture this build's kernels target, so
    /// the driver has no binary it can load. Reachable when the startup probe
    /// in `daemon::gpu_support` could not read the capability (no nvidia-smi,
    /// NVML mismatch) and so left the GPU enabled.
    ArchitectureTooOld,
    /// The weights fit, but nothing is left for the conversation cache, so the
    /// worker would come up and refuse every request — including a 42-token
    /// one (measured 2026-08-25). Detected by the loader rather than reported
    /// by the driver: see [`NO_KV_CACHE_ROOM_MARKER`].
    ///
    /// A retry on the same card requests the same bytes and gets the same
    /// answer, which is what makes it belong here; it stops being true the
    /// moment any GPU memory is freed, and the pool clears CPU pins then.
    NoKvCacheRoom,
}

/// The token the loader puts in its error when a GPU placement has no room left
/// for the conversation cache, and which [`permanent_gpu_failure`] matches to
/// send that model to the processor instead.
///
/// It is deliberately a fixed machine token rather than the human sentence
/// beside it. Matching prose is this codebase's recurring trap (#295) because
/// wording gets rewritten; a message crossing the worker IPC boundary loses its
/// type, so *something* in the text has to carry the meaning, and this is the
/// part that is ours and stable. Both sides use this constant — never a copy.
pub const NO_KV_CACHE_ROOM_MARKER: &str = "swarmllm:gpu-kv-budget-exhausted";

/// Classify a fatal worker error as permanently-GPU-fatal, if it is one.
///
/// `None` means "retry on the GPU is still reasonable" — the caller respawns
/// the worker as before rather than degrading the model to the CPU.
pub fn permanent_gpu_failure(message: &str) -> Option<PermanentGpuFailure> {
    let lower = message.to_ascii_lowercase();
    // The arch check runs first: CUDA_ERROR_NO_BINARY_FOR_GPU never co-occurs
    // with an OOM, but if a chained error mentioned both, the architecture is
    // the one that no amount of freed memory will fix.
    const ARCH_PATTERNS: &[&str] = &[
        // Driver API, as candle surfaces it.
        "no_binary_for_gpu",
        "no binary for gpu",
        // Runtime API wording for the same condition.
        "no kernel image is available",
        "invalid_ptx",
        "invalid ptx",
        "unsupported ptx version",
        "invalid device function",
    ];
    if ARCH_PATTERNS.iter().any(|p| lower.contains(p)) {
        return Some(PermanentGpuFailure::ArchitectureTooOld);
    }
    // Checked before the OOM patterns: this is our own marker for a condition
    // the driver never reports, and its text may legitimately mention memory.
    if message.contains(NO_KV_CACHE_ROOM_MARKER) {
        return Some(PermanentGpuFailure::NoKvCacheRoom);
    }
    const OOM_PATTERNS: &[&str] = &["out of memory", "out_of_memory", "outofmemory"];
    if OOM_PATTERNS.iter().any(|p| lower.contains(p)) {
        return Some(PermanentGpuFailure::OutOfMemory);
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A reply's grammar crosses the worker IPC although every serialized
    /// `SamplingParams` skips it (#336): `IpcGenerate::new` carries it beside,
    /// `rejoin_grammar` puts it back.
    #[test]
    fn a_generate_request_carries_its_reply_grammar_to_the_worker() {
        // A continued reply's: the text it starts after crosses with it.
        let grammar = std::sync::Arc::new(crate::types::ReplyGrammar {
            written: r#"{"a": 1, "#.into(),
            ..(*crate::inference::reply_grammar::settle_json_object()).clone()
        });
        let sampling = SamplingParams {
            reply_grammar: Some(grammar.clone()),
            ..Default::default()
        };
        let sent = IpcGenerate::new(
            Uuid::new_v4(),
            ModelId("m".into()),
            (0, 2),
            "hi".into(),
            sampling,
            None,
            true,
            false,
        );
        let mut got: IpcGenerate =
            serde_json::from_slice(&serde_json::to_vec(&sent).unwrap()).unwrap();
        assert!(
            got.sampling.reply_grammar.is_none(),
            "the premise: the sampling alone does not carry it"
        );
        got.rejoin_grammar();
        assert_eq!(got.sampling.reply_grammar.as_deref(), Some(&*grammar));
    }

    /// Write one frame by hand: a header this build may not know, and a payload.
    async fn write_raw_frame(w: &mut tokio::io::DuplexStream, header: &[u8], payload: &[u8]) {
        use tokio::io::AsyncWriteExt;
        w.write_all(&(header.len() as u32).to_le_bytes())
            .await
            .unwrap();
        w.write_all(header).await.unwrap();
        w.write_all(&(payload.len() as u32).to_le_bytes())
            .await
            .unwrap();
        w.write_all(payload).await.unwrap();
    }

    /// #150: a newer worker's message is skipped and the stream stays aligned —
    /// the message after it (with its payload) is read correctly. Before, the
    /// header failed to decode before its payload was read, the stream was lost
    /// and the worker evicted.
    #[tokio::test]
    async fn a_message_this_daemon_does_not_know_is_skipped_and_the_next_one_read() {
        let (mut worker, mut daemon) = tokio::io::duplex(1 << 16);
        write_raw_frame(&mut worker, br#"{"t":"SomethingNewer","x":1}"#, b"12345").await;
        send_worker(
            &mut worker,
            &WorkerMsg::CardAllocationProbe { took_ms: 7 },
            b"after",
        )
        .await
        .unwrap();
        let skipped = recv_worker(&mut daemon).await.unwrap_err();
        assert_eq!(unknown_worker_msg(&skipped), Some("SomethingNewer"));
        let (next, payload) = recv_worker(&mut daemon).await.unwrap();
        assert!(matches!(
            next,
            WorkerMsg::CardAllocationProbe { took_ms: 7 }
        ));
        assert_eq!(payload, b"after");
    }

    /// A KNOWN message that does not decode is a broken stream, not a newer
    /// worker: it stays fatal.
    #[tokio::test]
    async fn a_known_message_that_does_not_decode_is_still_fatal() {
        let (mut worker, mut daemon) = tokio::io::duplex(1 << 16);
        write_raw_frame(&mut worker, br#"{"t":"Token"}"#, b"").await;
        let e = recv_worker(&mut daemon).await.unwrap_err();
        assert_eq!(unknown_worker_msg(&e), None);
        assert_eq!(e.kind(), std::io::ErrorKind::InvalidData);
    }

    /// `WorkerMsgTag` names every `WorkerMsg` variant: this match stops
    /// compiling when a variant is added without a tag, and a tag is what serde
    /// writes as `t`.
    #[test]
    fn every_worker_message_has_a_tag() {
        fn tag(m: &WorkerMsg) -> WorkerMsgTag {
            match m {
                WorkerMsg::Ready => WorkerMsgTag::Ready,
                WorkerMsg::LayerResult(_) => WorkerMsgTag::LayerResult,
                WorkerMsg::BatchResult { .. } => WorkerMsgTag::BatchResult,
                WorkerMsg::Drafted { .. } => WorkerMsgTag::Drafted,
                WorkerMsg::PromptAdmitted { .. } => WorkerMsgTag::PromptAdmitted,
                WorkerMsg::PromptRefused { .. } => WorkerMsgTag::PromptRefused,
                WorkerMsg::Token { .. } => WorkerMsgTag::Token,
                WorkerMsg::GenerateDone { .. } => WorkerMsgTag::GenerateDone,
                WorkerMsg::CardAllocationProbe { .. } => WorkerMsgTag::CardAllocationProbe,
                WorkerMsg::ModelLoadTimed { .. } => WorkerMsgTag::ModelLoadTimed,
                WorkerMsg::ContextFits { .. } => WorkerMsgTag::ContextFits,
                WorkerMsg::CardFree { .. } => WorkerMsgTag::CardFree,
                WorkerMsg::Error { .. } => WorkerMsgTag::Error,
                WorkerMsg::PrefixManifestUpdate { .. } => WorkerMsgTag::PrefixManifestUpdate,
                WorkerMsg::PrefixFetchProbe { .. } => WorkerMsgTag::PrefixFetchProbe,
                WorkerMsg::PrefixSnapshotResponse { .. } => WorkerMsgTag::PrefixSnapshotResponse,
                WorkerMsg::Progress { .. } => WorkerMsgTag::Progress,
                WorkerMsg::Bye => WorkerMsgTag::Bye,
            }
        }
        for m in [
            WorkerMsg::Ready,
            WorkerMsg::Bye,
            WorkerMsg::CardAllocationProbe { took_ms: 1 },
            WorkerMsg::ModelLoadTimed {
                bytes: 1 << 30,
                ms: 2_500,
            },
            WorkerMsg::ContextFits {
                tokens: Some(4_608),
            },
            WorkerMsg::ContextFits { tokens: None },
            WorkerMsg::CardFree {
                free_bytes: Some(7_291 << 20),
            },
            WorkerMsg::CardFree { free_bytes: None },
            WorkerMsg::Drafted {
                request_id: Uuid::nil(),
                tokens: vec![],
                parents: vec![],
                scores: vec![],
                runner_ups: vec![],
            },
            WorkerMsg::PromptAdmitted {
                request_id: Uuid::nil(),
            },
            WorkerMsg::PromptRefused {
                request_id: Uuid::nil(),
                message: String::new(),
                layers_that_fit: 3,
            },
        ] {
            let written = serde_json::to_value(&m).unwrap()["t"].clone();
            assert_eq!(
                serde_json::from_value::<WorkerMsgTag>(written).unwrap(),
                tag(&m)
            );
        }
    }

    #[test]
    fn a_worker_reports_its_card_probe_only_to_a_daemon_that_said_it_reads_it() {
        // Unset = an older daemon (spawned the new binary after an update it
        // has not restarted into): it would lose the stream on the new variant.
        assert!(!daemon_reads_card_probe(None));
        assert!(daemon_reads_card_probe(Some(std::ffi::OsStr::new("1"))));
        assert!(!daemon_reads_card_probe(Some(std::ffi::OsStr::new("0"))));
        assert!(!daemon_reads_card_probe(Some(std::ffi::OsStr::new(""))));
    }

    #[test]
    fn cuda_oom_is_fatal() {
        // The exact shape candle produces on a CUDA OOM, as seen in the
        // 2026-07-21 bug report where the worker survived and kept 4456MB.
        assert!(worker_error_is_fatal(
            "Forward: Cuda(DriverError(CUDA_ERROR_OUT_OF_MEMORY, \"out of memory\"))"
        ));
        assert!(worker_error_is_fatal(
            "Inference error: CUDA_ERROR_OUT_OF_MEMORY"
        ));
        assert!(worker_error_is_fatal(
            "cuda error: an illegal memory access was encountered"
        ));
        assert!(worker_error_is_fatal(
            "cublas error: CUBLAS_STATUS_ALLOC_FAILED"
        ));
        assert!(worker_error_is_fatal(
            "Failed to allocate 4096 MB on device 0"
        ));
    }

    #[test]
    fn an_architecture_mismatch_is_fatal_and_permanent() {
        // What a pre-Ampere card produces once the build targets sm_80: the
        // context is created fine and module load is what fails. The worker
        // must be recycled AND the model moved off the GPU — respawning it
        // there reproduces this exactly, for every model, forever.
        let driver = "Forward: Cuda(DriverError(CUDA_ERROR_NO_BINARY_FOR_GPU, \
                      \"no kernel image is available for execution on the device\"))";
        assert!(worker_error_is_fatal(driver));
        assert_eq!(
            permanent_gpu_failure(driver),
            Some(PermanentGpuFailure::ArchitectureTooOld)
        );

        for msg in [
            "CUDA_ERROR_INVALID_PTX",
            "the provided PTX was compiled with an unsupported PTX version",
            "cudaErrorInvalidDeviceFunction: invalid device function",
            "no kernel image is available for execution on the device",
        ] {
            assert_eq!(
                permanent_gpu_failure(msg),
                Some(PermanentGpuFailure::ArchitectureTooOld),
                "input {msg:?}"
            );
        }
    }

    #[test]
    fn oom_is_permanent_but_classified_separately_from_arch() {
        // The two get different user-facing copy, so the classifier must not
        // collapse them — telling someone with a 2060 that they ran out of
        // memory would send them to free VRAM that was never the problem.
        assert_eq!(
            permanent_gpu_failure(
                "Forward: Cuda(DriverError(CUDA_ERROR_OUT_OF_MEMORY, \"out of memory\"))"
            ),
            Some(PermanentGpuFailure::OutOfMemory)
        );
    }

    #[test]
    fn a_transient_fatal_error_still_retries_on_the_gpu() {
        // Fatal (recycle the worker) but NOT permanent (the respawn may well
        // succeed). Pinning these to the CPU would silently halve throughput
        // for the rest of the run over a one-off fault.
        for msg in [
            "cuda error: an illegal memory access was encountered",
            "misaligned address",
            "device-side assert triggered",
        ] {
            assert!(worker_error_is_fatal(msg), "input {msg:?}");
            assert_eq!(permanent_gpu_failure(msg), None, "input {msg:?}");
        }
    }

    #[test]
    fn per_request_errors_are_not_fatal() {
        // These leave the loaded model perfectly usable — killing the worker
        // would throw away a multi-second model load for nothing.
        assert!(!worker_error_is_fatal("Tokenize: unknown token in prompt"));
        assert!(!worker_error_is_fatal(
            "Validation: temperature must be between 0.0 and 2.0"
        ));
        assert!(!worker_error_is_fatal("shape mismatch in stop sequence"));
        assert!(!worker_error_is_fatal("chat template evaluation failed"));
    }

    #[test]
    fn fatal_classification_is_case_insensitive() {
        assert!(worker_error_is_fatal("OUT OF MEMORY"));
        assert!(worker_error_is_fatal("Out Of Memory"));
        assert!(worker_error_is_fatal("out of memory"));
    }

    #[test]
    fn error_fatal_field_defaults_false_for_older_workers() {
        // Wire-compat: a worker built before the `fatal` field omits it.
        // Deserialization must not fail, and must not invent a `true`.
        let json =
            r#"{"t":"Error","request_id":"00000000-0000-0000-0000-000000000000","message":"boom"}"#;
        let msg: WorkerMsg = serde_json::from_str(json).unwrap();
        match msg {
            WorkerMsg::Error { fatal, message, .. } => {
                assert!(!fatal);
                assert_eq!(message, "boom");
            }
            other => panic!("expected Error, got {other:?}"),
        }
    }

    #[test]
    fn error_fatal_field_round_trips() {
        let msg = WorkerMsg::Error {
            local_memory_refusal: false,
            request_id: Uuid::nil(),
            message: "CUDA_ERROR_OUT_OF_MEMORY".into(),
            fatal: true,
        };
        let json = serde_json::to_string(&msg).unwrap();
        let back: WorkerMsg = serde_json::from_str(&json).unwrap();
        match back {
            WorkerMsg::Error { fatal, .. } => assert!(fatal),
            other => panic!("expected Error, got {other:?}"),
        }
    }

    #[test]
    fn cancel_request_round_trips() {
        let id = Uuid::new_v4();
        let json = serde_json::to_string(&DaemonMsg::CancelRequest { request_id: id }).unwrap();
        match serde_json::from_str::<DaemonMsg>(&json).unwrap() {
            DaemonMsg::CancelRequest { request_id } => assert_eq!(request_id, id),
            other => panic!("expected CancelRequest, got {other:?}"),
        }
    }

    #[test]
    fn spec_logits_roundtrip_empty() {
        let (bytes, dims) = encode_spec_logits(&[]);
        assert_eq!(bytes, Vec::<u8>::new());
        assert_eq!(dims, (0, 0));
    }

    #[test]
    fn spec_logits_roundtrip_small() {
        let original: Vec<Vec<f32>> = vec![
            vec![0.1, -0.2, 2.5, f32::NEG_INFINITY],
            vec![1.0, 2.0, 3.0, 4.0],
            vec![-0.0, 0.0, f32::MIN_POSITIVE, f32::MAX],
        ];
        let (bytes, dims) = encode_spec_logits(&original);
        assert_eq!(dims, (3, 4));
        assert_eq!(bytes.len(), 3 * 4 * 4);
        let restored = decode_spec_logits(&bytes, dims).unwrap();
        assert_eq!(restored.len(), original.len());
        for (o, r) in original.iter().zip(restored.iter()) {
            assert_eq!(o.len(), r.len());
            for (a, b) in o.iter().zip(r.iter()) {
                assert_eq!(a.to_bits(), b.to_bits(), "bit-exact roundtrip required");
            }
        }
    }

    #[test]
    fn spec_logits_decode_rejects_len_mismatch() {
        let err = decode_spec_logits(&[0u8; 3], (1, 2)).unwrap_err();
        assert!(err.contains("mismatch"));
    }

    #[test]
    fn vision_embeddings_len_elided_when_zero() {
        use crate::types::{ModelId, TensorFormat};
        use uuid::Uuid;
        let fwd = IpcForward {
            request_id: Uuid::nil(),
            sequence_num: 1,
            index_pos: 5,
            format: TensorFormat::FP32,
            model_id: ModelId("test".into()),
            layer_range: (0, 4),
            tp_meta: None,
            vision_embeddings_len: 0,
            requester_node_id: None,
            pre_embedded: false,
            sampling: Default::default(),
            adapter_id: None,
            draft_tokens: vec![],
            generated_ids: vec![],
            spec_logits_requested: false,
            spec_walk_at_tail: false,
            coupling_seed: None,
            stream_seq: None,
            truncate_kv_to: None,
            prompt_cache: None,
            for_the_owner: false,
            far_run: None,
            prompt_span: None,
            image_spans: Vec::new(),
            draft_parents: Vec::new(),
            draft_tree_from: None,
        };
        let json = serde_json::to_string(&fwd).unwrap();
        // When no vision payload, the field is elided to match pre-fix wire
        // shape and avoid bloating decode-hot-path forwards.
        assert!(
            !json.contains("vision_embeddings_len"),
            "expected vision_embeddings_len to be elided from JSON when zero, got: {json}"
        );
        assert!(!json.contains("vision_embeddings"));
    }

    /// The owner's segment of a split request reaches the worker marked as the
    /// owner's, a peer's never does, and a frame written before the field
    /// existed reads as the swarm's — never as the owner's.
    #[test]
    fn a_forward_says_whether_it_is_the_owners() {
        use crate::types::{ModelId, TensorFormat};
        use uuid::Uuid;
        let fwd = |for_the_owner| IpcForward {
            request_id: Uuid::nil(),
            sequence_num: 0,
            index_pos: 0,
            format: TensorFormat::FP32,
            model_id: ModelId("test".into()),
            layer_range: (0, 4),
            tp_meta: None,
            vision_embeddings_len: 0,
            requester_node_id: None,
            pre_embedded: false,
            sampling: Default::default(),
            adapter_id: None,
            draft_tokens: vec![],
            generated_ids: vec![],
            spec_logits_requested: false,
            spec_walk_at_tail: false,
            coupling_seed: None,
            stream_seq: None,
            truncate_kv_to: None,
            prompt_cache: None,
            for_the_owner,
            far_run: None,
            prompt_span: None,
            image_spans: Vec::new(),
            draft_parents: Vec::new(),
            draft_tree_from: None,
        };
        let owners = serde_json::to_string(&fwd(true)).unwrap();
        let back: IpcForward = serde_json::from_str(&owners).unwrap();
        assert!(
            back.for_the_owner,
            "lost on the way to the worker: {owners}"
        );
        let swarms = serde_json::to_string(&fwd(false)).unwrap();
        assert!(
            !swarms.contains("for_the_owner"),
            "a peer's forward carries nothing new"
        );
        let back: IpcForward = serde_json::from_str(&swarms).unwrap();
        assert!(!back.for_the_owner);
    }

    #[test]
    fn vision_embeddings_len_present_when_nonzero() {
        use crate::types::{ModelId, TensorFormat};
        use uuid::Uuid;
        let fwd = IpcForward {
            request_id: Uuid::nil(),
            sequence_num: 0,
            index_pos: 0,
            format: TensorFormat::FP32,
            model_id: ModelId("test".into()),
            layer_range: (0, 4),
            tp_meta: None,
            vision_embeddings_len: 12345,
            requester_node_id: None,
            pre_embedded: false,
            sampling: Default::default(),
            adapter_id: None,
            draft_tokens: vec![],
            generated_ids: vec![],
            spec_logits_requested: false,
            spec_walk_at_tail: false,
            coupling_seed: None,
            stream_seq: None,
            truncate_kv_to: None,
            prompt_cache: None,
            for_the_owner: false,
            far_run: None,
            prompt_span: None,
            image_spans: Vec::new(),
            draft_parents: Vec::new(),
            draft_tree_from: None,
        };
        let json = serde_json::to_string(&fwd).unwrap();
        assert!(json.contains("\"vision_embeddings_len\":12345"));
        let back: IpcForward = serde_json::from_str(&json).unwrap();
        assert_eq!(back.vision_embeddings_len, 12345);
    }

    #[test]
    fn spec_logits_realistic_vocab() {
        // γ=4, vocab=32000 — a TinyLlama-class spec verify payload.
        let mut original = Vec::with_capacity(5);
        for row_idx in 0..5 {
            let mut row = Vec::with_capacity(32000);
            for v in 0..32000 {
                row.push((row_idx as f32 * 0.01) + (v as f32 * 1e-6));
            }
            original.push(row);
        }
        let (bytes, dims) = encode_spec_logits(&original);
        assert_eq!(dims, (5, 32000));
        // Raw f32 binary = 5 * 32000 * 4 = 640_000 bytes — compare to the
        // old JSON encoding which would have been ~4-5 MB for the same
        // payload (each f32 ~7 bytes of decimal text).
        assert_eq!(bytes.len(), 640_000);
        let restored = decode_spec_logits(&bytes, dims).unwrap();
        assert_eq!(restored.len(), 5);
        assert_eq!(restored[0].len(), 32000);
        assert_eq!(restored[4][31999].to_bits(), original[4][31999].to_bits());
    }
}

#[cfg(test)]
mod no_kv_cache_room_tests {
    use super::*;

    /// A GPU placement with no conversation-cache room must be classified as
    /// permanently GPU-fatal, so the pool moves the model to the processor
    /// instead of leaving a worker up that refuses every request.
    #[test]
    fn a_zero_kv_budget_load_sends_the_model_to_the_cpu() {
        let msg = format!(
            "{NO_KV_CACHE_ROOM_MARKER} — this model needs about 3072 MB for a 8192-token \
             conversation cache, and the card has none left after its 2281 MB of weights"
        );
        assert_eq!(
            permanent_gpu_failure(&msg),
            Some(PermanentGpuFailure::NoKvCacheRoom)
        );
        // Both halves are needed: `classify_worker_error` only consults
        // `permanent_gpu_failure` inside its fatal branch, so a marker that is
        // not also fatal would never reach the CPU pin at all.
        assert!(worker_error_is_fatal(&msg));
    }

    /// The marker is checked before the generic memory patterns on purpose —
    /// its own message legitimately talks about memory, and misreading it as a
    /// driver OOM would log the wrong cause.
    #[test]
    fn the_marker_outranks_the_generic_memory_patterns() {
        let msg = format!("{NO_KV_CACHE_ROOM_MARKER} — out of memory for the cache");
        assert_eq!(
            permanent_gpu_failure(&msg),
            Some(PermanentGpuFailure::NoKvCacheRoom),
            "a driver OOM and a no-cache-room refusal are different causes and log differently"
        );
    }

    /// A real driver OOM must still classify as one — the new arm must not
    /// swallow the case it sits next to.
    #[test]
    fn a_driver_oom_is_still_an_oom() {
        let msg = "Forward: Cuda(DriverError(CUDA_ERROR_OUT_OF_MEMORY, \"out of memory\"))";
        assert_eq!(
            permanent_gpu_failure(msg),
            Some(PermanentGpuFailure::OutOfMemory)
        );
        assert!(worker_error_is_fatal(msg));
    }

    /// An ordinary failure must not be dragged onto the CPU by any of this.
    #[test]
    fn an_unrelated_failure_is_not_a_permanent_gpu_failure() {
        assert_eq!(
            permanent_gpu_failure("prompt exceeds the context window"),
            None
        );
        assert!(!worker_error_is_fatal("prompt exceeds the context window"));
    }
}
