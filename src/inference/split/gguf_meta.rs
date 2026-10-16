use std::collections::HashMap;
use std::io::Read as IoRead;
use std::path::Path;

use candle_core::quantized::gguf_file;

use crate::error::SwarmError;
use crate::inference::tokenizer::{SpecialTokenSpacing, SplitTokenizer};

/// Sidecar file carrying `token_embd.weight` for weight-tied models.
///
/// A node serving the LAST pipeline segment needs the output head, but for a
/// weight-tied model that tensor physically lives in shard 0 — which that node
/// often does not hold. This file carries the raw tensor bytes so the head can
/// be loaded without shard 0. Written by `extract_sidecar_tensors` and
/// `download_sidecar_tensors`; read back by `ShardReader`.
pub const TIED_OUTPUT_FILENAME: &str = "tied_output_weight.bin";

/// Sidecar file carrying `rope_freqs.weight` — Llama 3.1/3.2's per-dimension
/// RoPE frequency divisors, 256 bytes. It lives in shard 0 like the tied head,
/// and EVERY layer's rotation needs it, so a node serving later layers without
/// shard 0 would otherwise rotate with plain RoPE and drift from the model the
/// longer the conversation (#124).
pub const ROPE_FREQS_FILENAME: &str = "rope_freqs.bin";

/// The GGUF tensor carried by [`ROPE_FREQS_FILENAME`].
pub const ROPE_FREQS_TENSOR: &str = "rope_freqs.weight";

/// One tensor a node may need although it lives in shard 0, and the sidecar
/// file that carries it for a node without shard 0.
pub struct SidecarSpec<'a> {
    /// File name inside the model directory.
    pub filename: &'static str,
    /// What it is, for logs and errors.
    pub what: &'static str,
    /// The GGUF tensor it carries — also how it is found in a PACKED shard's
    /// own tensor table, where its offset is not its GGUF offset.
    pub tensor: &'static str,
    /// Where the tensor sits in the GGUF.
    pub location: &'a TensorLocation,
}

/// Parse a GGUF header from a file on disk.
///
/// **The single way to read a GGUF header off a path**, and the buffering is
/// the whole reason it exists. `gguf_file::Content::read` walks the metadata
/// with many tiny reads — for every string, a length and then its bytes — so
/// handing it a bare `File` turns each of those into a syscall. A 7.8 MB header
/// carrying a 128k-token vocabulary and 280k merges is roughly 820k reads.
///
/// Measured on the live node 2026-08-29 (gotcha #410): `GET /api/admin/models`
/// took 11.2 s of which **9.6 s was kernel time** — the request parses every
/// local model's header, and seven call sites were passing an unbuffered
/// `File`. Optimisation cannot touch that cost, which is why the release binary
/// was no faster than a debug one on this path; only buffering removes it.
/// Direct comparison on one 7.8 MB header: 980 ms unbuffered against 98 ms
/// buffered.
///
/// The capacity is deliberately larger than `BufReader`'s 8 KB default: headers
/// run to several MB, and at 1 MB a whole one costs single-digit syscalls. It
/// is transient — freed when this returns.
///
/// Call sites that already hold the bytes in memory (a `Cursor` over a slice or
/// an mmap) do not need this and must not pay for a second copy;
/// `a_gguf_header_is_never_parsed_straight_off_an_unbuffered_file` in
/// `tests/repo_consistency.rs` recognises exactly those.
pub fn read_gguf_header(path: &Path) -> Result<gguf_file::Content, SwarmError> {
    let file = std::fs::File::open(path).map_err(SwarmError::Io)?;
    let mut reader = std::io::BufReader::with_capacity(GGUF_HEADER_READ_BUFFER_BYTES, file);
    gguf_file::Content::read(&mut reader).map_err(|e| {
        SwarmError::Internal(format!(
            "Failed to read GGUF header: {}",
            explain_gguf_parse_error(&e)
        ))
    })
}

/// What a failure to parse a GGUF header means, in words a person can act on.
///
/// The case that needs it: a file whose tensors use a quantization type this
/// engine does not implement. candle refuses it as `unknown dtype for tensor
/// 18`, which says nothing to someone who picked `IQ3_XXS` off a model page —
/// and the popular MoE uploads (Qwen3-30B-A3B's smallest builds) are exactly
/// the `IQ*` family. Every other failure is passed through as it was.
pub fn explain_gguf_parse_error(e: &candle_core::Error) -> String {
    let text = e.to_string();
    let Some(code) = text
        .split("unknown dtype for tensor ")
        .nth(1)
        .and_then(|rest| rest.split(|c: char| !c.is_ascii_digit()).next())
        .and_then(|n| n.parse::<u32>().ok())
    else {
        return text;
    };
    // ggml's type numbers (`ggml.h`, `enum ggml_type`) for the ones people meet.
    let name = match code {
        16 => "IQ2_XXS",
        17 => "IQ2_XS",
        18 => "IQ3_XXS",
        19 => "IQ1_S",
        20 => "IQ4_NL",
        21 => "IQ3_S",
        22 => "IQ2_S",
        23 => "IQ4_XS",
        29 => "IQ1_M",
        34 => "TQ1_0",
        35 => "TQ2_0",
        39 => "MXFP4",
        _ => "a type this build does not know",
    };
    format!(
        "this GGUF stores its weights as {name} (ggml type {code}), a quantization this \
         engine cannot read. Choose a K-quant build of the model instead — Q2_K, Q3_K_M, \
         Q4_K_M, Q5_K_M, Q6_K or Q8_0"
    )
}

/// Read-ahead window for [`read_gguf_header`]. See that function for why.
const GGUF_HEADER_READ_BUFFER_BYTES: usize = 1 << 20;

/// Extract the GGUF `general.architecture` string, defaulting to `"llama"` when absent.
pub fn gguf_arch_str(ct: &gguf_file::Content) -> String {
    ct.metadata
        .get("general.architecture")
        .and_then(|v| v.to_string().ok().cloned())
        .unwrap_or_else(|| "llama".to_string())
}

/// Is layer `layer_idx` a RECURRENT layer — Qwen 3.5's Gated DeltaNet, which
/// carries a fixed-size state instead of a KV cache that grows per position?
///
/// Read off the tensor table: the loader builds a DeltaNet layer exactly where
/// `blk.N.ssm_alpha.weight` exists, so this is the loader's own test, and a
/// header answers it for a segment whose parts are not on disk. llama.cpp
/// derives the same layers from `full_attention_interval`
/// (`(il + 1) % interval != 0`); the file is the stronger witness.
pub(crate) fn layer_is_recurrent(ct: &gguf_file::Content, layer_idx: usize) -> bool {
    ct.tensor_infos
        .contains_key(&recurrent_layer_marker(layer_idx))
}

/// A model's sliding-window attention, read as llama.cpp reads it — the ONE
/// reading: the loader sets each layer's `LayerWeights::sliding_window` from
/// [`Self::for_layer`].
///
/// Only the families whose window llama.cpp APPLIES and this build runs.
/// Gemma 2 (`llama_model_gemma2::load_arch_hparams`): every other layer from
/// layer 0, 4,096 positions unless the file says otherwise. Read without it,
/// every Gemma 2 layer attended the whole conversation past 4,096 tokens, and
/// its replies drifted from llama.cpp's there (FUTURE_WORK #278). Gemma 3
/// (`llama_model_gemma3`): five layers in six, only when the file names a
/// window above 0 — 512 on the 1B, 1,024 from the 4B up. Phi-3
/// declares a window that llama.cpp switches OFF (`llama_model_phi3`, "Phi SWA
/// is currently disabled", ggml-org/llama.cpp#13676), so it is not read here
/// either: a reply is judged against llama.cpp. Llama 4's chunked attention
/// is a different rule (`LLAMA_SWA_TYPE_CHUNKED`), not this one.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SlidingWindow {
    /// Positions a windowed layer's query sees, its own included.
    pub(crate) size: usize,
    pattern: SwaPattern,
}

/// Does this node attend a sliding window where a model has one —
/// `SWARMLLM_SLIDING_WINDOW=0` is the A/B arm that does not
/// ([`SlidingWindow::read`]).
pub(crate) fn windows_attended() -> bool {
    std::env::var("SWARMLLM_SLIDING_WINDOW").as_deref() != Ok("0")
}

/// Which layers are windowed: llama.cpp's
/// `{arch}.attention.sliding_window_pattern`, one flag per layer or a period
/// (`llama_model_base::load_swa_pattern`), else the family's own period.
#[derive(Debug, Clone, PartialEq, Eq)]
enum SwaPattern {
    /// Layer `il` is windowed when `il % n < n - 1`, every layer when `n` is
    /// 0 (`llama_hparams::set_swa_pattern`, no dense layer first).
    Period(usize),
    Layers(Vec<bool>),
}

impl SlidingWindow {
    /// The window `ct` describes, or `None` for a model with none.
    /// `SWARMLLM_SLIDING_WINDOW=0` reads none — the A/B arm, every layer
    /// attending the whole conversation as before #278.
    pub(crate) fn read(ct: &gguf_file::Content) -> Option<Self> {
        if !windows_attended() {
            return None;
        }
        Self::from_file(ct)
    }

    /// The window as the file and family describe it, whatever the A/B arm
    /// says — for what follows the layer pattern rather than the masking
    /// (`split::rope::LayerRope`, the windowed layers' own rotation).
    pub(crate) fn from_file(ct: &gguf_file::Content) -> Option<Self> {
        let arch = gguf_arch_str(ct);
        // (the window when the file names none, the period)
        let (default_size, period) = match arch.as_str() {
            "gemma2" => (4096, 2),
            "gemma3" | "gemma4" => (0, 6),
            _ => return None,
        };
        let size = ct
            .metadata
            .get(&format!("{arch}.attention.sliding_window"))
            .and_then(|v| v.to_u32().ok())
            .map_or(default_size, |v| v as usize);
        let pattern = match ct
            .metadata
            .get(&format!("{arch}.attention.sliding_window_pattern"))
        {
            Some(gguf_file::Value::Array(flags)) => {
                SwaPattern::Layers(flags.iter().map(|f| f.to_bool().unwrap_or(false)).collect())
            }
            Some(v) => SwaPattern::Period(v.to_u32().map_or(period, |n| n as usize)),
            None => SwaPattern::Period(period),
        };
        (size > 0).then_some(Self { size, pattern })
    }

    /// The window of layer `il` (its index in the WHOLE model), or `None`
    /// for a layer that attends the whole conversation.
    pub(crate) fn for_layer(&self, il: usize) -> Option<usize> {
        let windowed = match &self.pattern {
            SwaPattern::Period(0) => true,
            SwaPattern::Period(n) => il % n < n - 1,
            SwaPattern::Layers(flags) => flags.get(il).copied().unwrap_or(false),
        };
        windowed.then_some(self.size)
    }
}

/// One layer's attention: its K/V heads, its head width and how much of the
/// head RoPE turns.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct AttentionShape {
    pub(crate) n_kv_head: usize,
    pub(crate) head_dim: usize,
    pub(crate) rope_dim: usize,
}

impl AttentionShape {
    /// K or V elements this layer keeps per position.
    pub(crate) fn kv_width(&self) -> usize {
        self.n_kv_head * self.head_dim
    }
}

/// Every layer's [`AttentionShape`], read as llama.cpp reads it for every
/// family (`llama_hparams::n_head_kv(il)`, `n_embd_head_k(il)`, `n_rot(il)`):
/// `attention.head_count_kv` is one count or one per layer, and a windowed
/// layer ([`SlidingWindow`]) takes `attention.key_length_swa` and
/// `rope.dimension_count_swa` where the file names them. Gemma 4 does: 256-wide
/// heads on its windowed layers against 512 on its global ones, and 8 K/V heads
/// against 1 on the 12B. A file naming neither gets one shape for every layer —
/// the model-wide reading every release before took.
pub(crate) fn attention_shapes(ct: &gguf_file::Content) -> Result<Vec<AttentionShape>, SwarmError> {
    let arch = gguf_arch_str(ct);
    let key = |suffix: &str| ct.metadata.get(&format!("{arch}.{suffix}"));
    let u32_of = |suffix: &str| {
        key(suffix)
            .and_then(|v| v.to_u32().ok())
            .map(|v| v as usize)
    };
    let bad = |what: &str| SwarmError::Inference(format!("GGUF metadata error: {arch}.{what}"));
    let n_layer = u32_of("block_count").ok_or_else(|| bad("block_count is missing"))?;
    let n_head =
        u32_of("attention.head_count").ok_or_else(|| bad("attention.head_count is missing"))?;
    let embd = u32_of("embedding_length").ok_or_else(|| bad("embedding_length is missing"))?;
    // A count stored as any integer type: llama.cpp's `get_arr` takes signed
    // and unsigned alike, and gguf-py writes a Python list as INT32.
    let count = |v: &gguf_file::Value| match *v {
        gguf_file::Value::U8(n) => Some(n as usize),
        gguf_file::Value::U16(n) => Some(n as usize),
        gguf_file::Value::U32(n) => Some(n as usize),
        gguf_file::Value::U64(n) => usize::try_from(n).ok(),
        gguf_file::Value::I8(n) => usize::try_from(n).ok(),
        gguf_file::Value::I16(n) => usize::try_from(n).ok(),
        gguf_file::Value::I32(n) => usize::try_from(n).ok(),
        gguf_file::Value::I64(n) => usize::try_from(n).ok(),
        _ => None,
    };
    let n_kv: Vec<usize> = match key("attention.head_count_kv") {
        Some(gguf_file::Value::Array(per_layer)) => {
            let counts: Vec<usize> = per_layer
                .iter()
                .map(count)
                .collect::<Option<_>>()
                .ok_or_else(|| bad("attention.head_count_kv holds a value that is not a count"))?;
            if counts.len() < n_layer {
                return Err(bad(
                    "attention.head_count_kv names fewer layers than block_count",
                ));
            }
            counts
        }
        Some(v) => {
            vec![count(v).ok_or_else(|| bad("attention.head_count_kv is not a count"))?; n_layer]
        }
        None => return Err(bad("attention.head_count_kv is missing")),
    };
    let head_dim = u32_of("attention.key_length").unwrap_or(embd / n_head.max(1));
    let rope_dim = u32_of("rope.dimension_count").unwrap_or(head_dim);
    let head_dim_swa = u32_of("attention.key_length_swa").unwrap_or(head_dim);
    let rope_dim_swa = u32_of("rope.dimension_count_swa").unwrap_or(rope_dim);
    let window = SlidingWindow::from_file(ct);
    Ok((0..n_layer)
        .map(|il| {
            let windowed = window.as_ref().is_some_and(|w| w.for_layer(il).is_some());
            AttentionShape {
                n_kv_head: n_kv[il],
                head_dim: if windowed { head_dim_swa } else { head_dim },
                rope_dim: if windowed { rope_dim_swa } else { rope_dim },
            }
        })
        .collect())
}

/// What a layer's queries are multiplied by so that the attention's own
/// `1/sqrt(head_dim)` leaves the family's scale (`LayerWeights::query_scale`),
/// or `None` where the two agree; `head_dim` is the layer's own
/// ([`AttentionShape`]). Gemma 4 scales by nothing at all — its queries and
/// keys are already normalised per head (llama.cpp's `f_attention_scale =
/// 1.0`, "no pre-attn scaling"), so its queries are multiplied by
/// `sqrt(head_dim)`. Read as llama.cpp reads it: Gemma 2's 27B —
/// the size llama.cpp knows by its 46 layers — scales by
/// `1/sqrt(n_embd / n_head)` (`llama_model_gemma2::load_arch_hparams`, after
/// gemma_pytorch `config.py`), every other size by `1/sqrt(head_dim)`; Gemma
/// 3's 27B (62 layers) the same way (`llama_model_gemma3`). On the
/// 27B that is 1/sqrt(144) against 1/sqrt(128): read without it, every
/// attention score was 6% too large (FUTURE_WORK #279).
/// `SWARMLLM_QUERY_SCALE=0` reads none — the A/B arm.
pub(crate) fn query_scale(ct: &gguf_file::Content, head_dim: usize) -> Option<f64> {
    if std::env::var("SWARMLLM_QUERY_SCALE").as_deref() == Ok("0") {
        return None;
    }
    let arch = gguf_arch_str(ct);
    if arch == "gemma4" {
        return Some((head_dim as f64).sqrt());
    }
    let read = |key: &str| {
        ct.metadata
            .get(&format!("{arch}.{key}"))
            .and_then(|v| v.to_u32().ok())
            .map(|v| v as usize)
    };
    let scaled_by_width = match arch.as_str() {
        "gemma2" => read("block_count") == Some(46),
        "gemma3" => read("block_count") == Some(62),
        _ => false,
    };
    if !scaled_by_width {
        return None;
    }
    let (n_embd, n_head) = (read("embedding_length")?, read("attention.head_count")?);
    // llama.cpp divides in integers before the square root.
    let width = n_embd.checked_div(n_head)?;
    let head_dim = read("attention.key_length").unwrap_or(width);
    (width != head_dim && width > 0).then(|| (head_dim as f64 / width as f64).sqrt())
}

/// The tensor whose presence marks layer `layer_idx` as recurrent — the ONE
/// spelling, for the header ([`layer_is_recurrent`]) and the manifest's
/// tensor table ([`GgufTensorMeta::layers_keeping_kv`]) alike.
fn recurrent_layer_marker(layer_idx: usize) -> String {
    format!("blk.{layer_idx}.ssm_alpha.weight")
}

/// Which of layers `[start, end)` keep a KV cache: all of them for a
/// transformer, the full-attention ones only for a model that interleaves
/// recurrent layers ([`layer_is_recurrent`]). The ONE answer for everything
/// that charges a memory for conversations — the loader's KV budget and every
/// estimate before a load, through `kv_budget::KvLayout`. Charging every layer
/// of Qwen 3.5 3B-9B as attention overstated it fourfold: ~805 MB at the
/// admission context for the 9B, the margin by which it missed an 8 GB card
/// (FUTURE_WORK #228).
pub(crate) fn kv_layers(
    ct: &gguf_file::Content,
    start: usize,
    end: usize,
) -> impl Iterator<Item = usize> + '_ {
    (start..end).filter(move |&layer| !layer_is_recurrent(ct, layer))
}

/// The bytes of layers `[start, end)`'s routed expert stacks
/// (`ffn_{gate,up,down}_exps`, or Gemma 4's fused `ffn_gate_up_exps`), 0 for
/// a dense layer — what keeping their
/// experts in the machine's memory takes off the card (#138 (3)). The ONE
/// answer for the loader, which places them, and the pool, which chooses how
/// many and charges each device for them.
pub(crate) fn routed_expert_bytes(ct: &gguf_file::Content, start: usize, end: usize) -> u64 {
    (start..end)
        .flat_map(|layer| {
            [
                "ffn_gate_exps",
                "ffn_up_exps",
                "ffn_gate_up_exps",
                "ffn_down_exps",
            ]
            .map(|t| format!("blk.{layer}.{t}.weight"))
        })
        .filter_map(|name| ct.tensor_infos.get(&name))
        .map(tensor_byte_size)
        .fold(0u64, u64::saturating_add)
}

/// Metadata extracted from GGUF header, stored in manifest for all nodes.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct GgufTensorMeta {
    /// Tensor name → (offset from tensor_data_start, size in bytes, dtype tag).
    pub tensors: HashMap<String, TensorLocation>,
    /// Offset in the GGUF file where tensor data begins.
    pub tensor_data_offset: u64,
    /// Friendly model name from GGUF `general.name` metadata.
    pub model_name: Option<String>,
    /// Model hyperparameters extracted from GGUF metadata.
    pub head_count: usize,
    pub head_count_kv: usize,
    pub block_count: usize,
    pub embedding_length: usize,
    /// Per-head dimension. Prefers `<arch>.attention.key_length` from GGUF
    /// (Qwen3 uses 128 vs embed/heads=64); falls back to `embedding_length /
    /// head_count`. `serde(default)` so older manifests still deserialize.
    #[serde(default)]
    pub head_dim: usize,
    pub rope_dim: usize,
    pub rope_freq_base: f32,
    pub rms_norm_eps: f64,
    /// DeepSeek-V2/V3 expert count (0 for non-MoE models).
    #[serde(default)]
    pub expert_count: usize,
    /// Experts each token is routed to (`<arch>.expert_used_count`), 0 for a
    /// dense model or a header that does not say. What a token READS of a
    /// mixture-of-experts layer: [`Self::active_bytes_per_layer`].
    #[serde(default)]
    pub expert_used_count: usize,
    /// Raw GGUF architecture string (e.g. "llama", "qwen2", "qwen35").
    #[serde(default)]
    pub architecture: String,
    /// The context the model DECLARES (`<arch>.context_length`), 0 when the
    /// header does not say. What a given node serves is this capped by that
    /// node's ceiling — `split::effective_context_length_with` — so this is the
    /// half a coordinator can know about a peer from the header alone.
    #[serde(default)]
    pub context_length: usize,
    /// How the model pools its hidden states into ONE vector
    /// (`<arch>.pooling_type`: 1 mean, 2 cls, 3 last, 4 rank), 0 when the
    /// header does not say — every generative model. A non-zero value is an
    /// embedding model: [`Self::is_embedding_model`].
    #[serde(default)]
    pub pooling_type: u32,
    /// How each layer keeps a conversation's cache (`kv_budget::KvLayout`),
    /// read off the header with the rest — what a coordinator prices a
    /// peer's room for a prompt by (`scheduler::prompt_kv_bytes_per_layer`):
    /// each layer by its own shape, a windowed one by its window where the
    /// peer's caches slide (#281 (4)). Never carried in a serialized copy;
    /// `None` there, and the bound charges the widest layer at every position.
    #[serde(skip)]
    pub(crate) kv_layout: Option<super::kv_budget::KvLayout>,
}

/// Why this build will not run a model to write replies, read off its header.
/// The one answer the loader, the router and auto-manage all act on
/// ([`GgufTensorMeta::refusal`], `SharedState::refused_model`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ModelRefusal {
    /// A family this build has no loader for (`ModelArch::is_supported`).
    Architecture(String),
    /// An embedding model: it turns text into one vector for search and has
    /// no head trained to predict the next token. Run as a chat model it
    /// repeats a word for as long as it is allowed to (field report
    /// 2026-10-15: Qwen3-Embedding-0.6B, 30 s of "IP address").
    EmbeddingModel,
}

impl ModelRefusal {
    /// The error a request for this model is refused with — 400, because no
    /// node and no retry can make this model answer a chat.
    pub fn into_error(self) -> SwarmError {
        match self {
            Self::Architecture(arch) => SwarmError::Validation(format!(
                "Unsupported model architecture '{arch}'. Supported architectures: {}",
                super::ModelArch::supported_list().join(", ")
            )),
            Self::EmbeddingModel => SwarmError::Validation(
                "This model is an embedding model: it turns text into numbers for search \
                 tools and cannot write replies. Choose a chat model instead."
                    .into(),
            ),
        }
    }

    /// The tag the model list carries (`cannot_chat`).
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Architecture(_) => "unsupported_architecture",
            Self::EmbeddingModel => "embedding_model",
        }
    }
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct TensorLocation {
    /// Byte offset relative to tensor_data_offset.
    pub offset: u64,
    /// Total size in bytes.
    pub size: u64,
}

/// How many bytes of tensor data a header's entry describes — the size every
/// tensor table is cut by (`TensorLocation::size`, and through it
/// `compute_layer_shard_layouts`), so the shard loader compares a table
/// against its header with this same computation.
///
/// Checked arithmetic, because a header can arrive from a peer: a crafted
/// shape must not overflow. `elem_count` is capped at 2^40 (~1 trillion) — no
/// legitimate tensor exceeds it — and an impossible entry answers 0.
pub(crate) fn tensor_byte_size(info: &candle_core::quantized::gguf_file::TensorInfo) -> u64 {
    let block_size = info.ggml_dtype.block_size();
    let elem_count = info.shape.elem_count();
    const MAX_ELEM_COUNT: usize = 1 << 40;
    if block_size == 0 || elem_count > MAX_ELEM_COUNT {
        return 0;
    }
    info.ggml_dtype
        .type_size()
        .checked_mul(elem_count)
        .map(|v| (v / block_size) as u64)
        .unwrap_or(0)
}

impl GgufTensorMeta {
    /// How many of [`kv_layers`] there are, read off the manifest's tensor table — for a
    /// node that prices a model from its manifest rather than its header (the
    /// scheduler's peer bound, `kv_bytes_per_position_per_layer`).
    pub(crate) fn layers_keeping_kv(&self, start: usize, end: usize) -> usize {
        (start..end)
            .filter(|&layer| !self.tensors.contains_key(&recurrent_layer_marker(layer)))
            .count()
    }

    /// Location of the tensor doubling as the output head on a weight-tied
    /// model, or `None` when the model ships a separate `output.weight`.
    ///
    /// Weight tying means reusing `token_embd.weight` as the LM head, so the
    /// GGUF carries no `output.weight` at all. This is the single definition of
    /// "is this model weight-tied" — the two sidecar writers
    /// (`extract_sidecar_tensors`, `download_sidecar_tensors`) and the
    /// reader (`ShardReader`) all consult it, so a producer can never disagree
    /// with the consumer about which tensor the sidecar holds.
    pub fn tied_output_location(&self) -> Option<&TensorLocation> {
        if self.tensors.contains_key("output.weight") {
            return None;
        }
        self.tensors.get("token_embd.weight")
    }

    /// Every shard-0 tensor this model has that a node WITHOUT shard 0 still
    /// needs: the output head of a weight-tied model, and the RoPE frequency
    /// factors of a model that ships them. **The single list** the producers
    /// (`extract_sidecar_tensors`, `download_sidecar_tensors`) and the reader
    /// (`resolve_sidecars`) all walk, so adding one here reaches all three — the
    /// tied head was the only one until 2026-09-27, and the factors' absence
    /// cost every Llama 3 model its long-context accuracy (#124).
    pub fn sidecar_tensors(&self) -> Vec<SidecarSpec<'_>> {
        let mut out = Vec::new();
        if let Some(location) = self.tied_output_location() {
            out.push(SidecarSpec {
                filename: TIED_OUTPUT_FILENAME,
                what: "tied output head (token_embd.weight)",
                tensor: "token_embd.weight",
                location,
            });
        }
        if let Some(location) = self.tensors.get(ROPE_FREQS_TENSOR) {
            out.push(SidecarSpec {
                filename: ROPE_FREQS_FILENAME,
                what: "RoPE frequency factors (rope_freqs.weight)",
                tensor: ROPE_FREQS_TENSOR,
                location,
            });
        }
        out
    }

    /// Extract tensor metadata from a GGUF file header on disk.
    /// Only needs to read the header, not the full file.
    pub fn from_gguf_file(path: &Path) -> Result<Self, SwarmError> {
        let ct = read_gguf_header(path)?;
        Self::from_content(&ct)
    }

    /// Extract tensor metadata from an already-parsed GGUF `Content`.
    /// Supports multiple architecture prefixes (llama, qwen2, mistral, etc.).
    pub fn from_content(ct: &gguf_file::Content) -> Result<Self, SwarmError> {
        let model_name = ct
            .metadata
            .get("general.name")
            .and_then(|v| v.to_string().ok().cloned());

        let arch = gguf_arch_str(ct);

        let md_get = |suffix: &str| {
            let key = format!("{arch}.{suffix}");
            ct.metadata
                .get(&key)
                .ok_or_else(|| SwarmError::Internal(format!("Missing GGUF metadata: {key}")))
        };
        let md_u32 = |suffix: &str| -> Result<usize, SwarmError> {
            Ok(md_get(suffix)?
                .to_u32()
                .map_err(|e| SwarmError::Internal(format!("Bad metadata: {e}")))?
                as usize)
        };

        let head_count = md_u32("attention.head_count")?;
        if head_count == 0 {
            return Err(SwarmError::Inference(
                "GGUF metadata error: attention.head_count is zero".into(),
            ));
        }
        // Sanity caps on peer-supplied GGUF dimensions. Without these a crafted
        // GGUF file (downloaded from network or from HF) can drive the worker
        // into oversized KV cache / mask allocations and OOM the subprocess.
        // Limits chosen well above any current real architecture:
        // - block_count (= layer count): 256 is 2× DeepSeek's 128
        // - embedding_length: 65 536 is 4× any current 70B model's hidden dim
        // - head_count: 256 (Llama-3 70B has 64)
        const MAX_BLOCK_COUNT: usize = 256;
        const MAX_EMBEDDING_LENGTH: usize = 65_536;
        const MAX_HEAD_COUNT: usize = 256;
        if head_count > MAX_HEAD_COUNT {
            return Err(SwarmError::Inference(format!(
                "GGUF metadata error: attention.head_count={head_count} exceeds cap {MAX_HEAD_COUNT}"
            )));
        }
        let block_count = md_u32("block_count")?;
        if block_count > MAX_BLOCK_COUNT {
            return Err(SwarmError::Inference(format!(
                "GGUF metadata error: block_count={block_count} exceeds cap {MAX_BLOCK_COUNT}"
            )));
        }
        // Next-token-prediction (MTP) blocks are stored as extra decoder blocks
        // after the model's own and are NOT run by an ordinary forward —
        // llama.cpp: `n_layer() = n_layer_all - n_layer_nextn`, "loaded as extra
        // decoder blocks but not executed in the main pass" (`models/qwen35.cpp`).
        // Every planner and loader reads this count as the layers to run.
        let block_count = block_count.saturating_sub(md_u32("nextn_predict_layers").unwrap_or(0));
        let embedding_length = md_u32("embedding_length")?;
        if embedding_length == 0 || embedding_length > MAX_EMBEDDING_LENGTH {
            return Err(SwarmError::Inference(format!(
                "GGUF metadata error: embedding_length={embedding_length} out of range (1..={MAX_EMBEDDING_LENGTH})"
            )));
        }
        // The KV shape of the WIDEST layer (`attention_shapes`): one shape for
        // most families; Gemma 4's windowed and global layers differ, and every
        // estimate that charges each layer this one shape is then an upper
        // bound (9-14% over on the real files) rather than an undercount.
        // head_dim: attention.key_length when named (Qwen3 uses 128 vs embed/heads=64).
        let widest = attention_shapes(ct)?
            .into_iter()
            .max_by_key(AttentionShape::kv_width)
            .ok_or_else(|| SwarmError::Inference("GGUF metadata error: no layers".into()))?;
        let (head_count_kv, head_dim) = (widest.n_kv_head, widest.head_dim);
        if head_count_kv > MAX_HEAD_COUNT {
            return Err(SwarmError::Inference(format!(
                "GGUF metadata error: attention.head_count_kv={head_count_kv} exceeds cap {MAX_HEAD_COUNT}"
            )));
        }
        // rope.dimension_count may not exist for all architectures — derive from
        // head_dim. The global layers' (a windowed layer's own is in its shape).
        let rope_dim = md_get("rope.dimension_count")
            .and_then(|v| v.to_u32().map_err(SwarmError::internal))
            .unwrap_or_else(|_| {
                ct.metadata
                    .get(&format!("{arch}.attention.key_length"))
                    .and_then(|v| v.to_u32().ok())
                    .unwrap_or((embedding_length / head_count) as u32)
            }) as usize;
        let rms_norm_eps = md_get("attention.layer_norm_rms_epsilon")?
            .to_f32()
            .map_err(|e| SwarmError::Internal(format!("Bad metadata: {e}")))?
            as f64;
        let rope_freq_base = ct
            .metadata
            .get(&format!("{arch}.rope.freq_base"))
            .and_then(|v| v.to_f32().ok())
            .unwrap_or(10000f32);

        let mut tensors = HashMap::new();
        for (name, info) in &ct.tensor_infos {
            tensors.insert(
                name.clone(),
                TensorLocation {
                    offset: info.offset,
                    size: tensor_byte_size(info),
                },
            );
        }

        // Read expert count for DeepSeek-V2/V3 models
        let expert_count = ct
            .metadata
            .get(&format!("{arch}.expert_count"))
            .and_then(|v| v.to_u32().ok())
            .unwrap_or(0) as usize;
        let expert_used_count = ct
            .metadata
            .get(&format!("{arch}.expert_used_count"))
            .and_then(|v| v.to_u32().ok())
            .unwrap_or(0) as usize;
        let context_length = ct
            .metadata
            .get(&format!("{arch}.context_length"))
            .and_then(|v| v.to_u32().ok())
            .unwrap_or(0) as usize;
        // Ollama's rule (`server/images.go`, `ggufCapabilities`): a GGUF that
        // names a pooling type is an embedding model, not a completion one.
        // llama.cpp's converter writes the key only for embedding models;
        // NONE (0) is read as "not said", so a generative file that spells it
        // out still loads.
        let pooling_type = ct
            .metadata
            .get(&format!("{arch}.pooling_type"))
            .and_then(|v| v.to_u32().ok())
            .unwrap_or(0);

        let mut meta = GgufTensorMeta {
            tensors,
            tensor_data_offset: ct.tensor_data_offset,
            model_name,
            head_count,
            head_count_kv,
            block_count,
            embedding_length,
            head_dim,
            rope_dim,
            rope_freq_base,
            rms_norm_eps,
            expert_count,
            expert_used_count,
            architecture: arch,
            context_length,
            pooling_type,
            kv_layout: None,
        };
        meta.kv_layout =
            super::kv_budget::KvLayout::read(ct, &meta, meta.context_length.max(1)).ok();
        Ok(meta)
    }

    /// An embedding model — see [`Self::pooling_type`].
    pub fn is_embedding_model(&self) -> bool {
        self.pooling_type != 0
    }

    /// Why this build will not run this model to write replies, or `None`.
    pub fn refusal(&self) -> Option<ModelRefusal> {
        if !super::ModelArch::from_gguf_arch(&self.architecture).is_supported() {
            return Some(ModelRefusal::Architecture(self.architecture.clone()));
        }
        if let Some(part) = self.unbuilt_gemma4_part() {
            return Some(ModelRefusal::Architecture(format!("gemma4 with {part}")));
        }
        self.is_embedding_model()
            .then_some(ModelRefusal::EmbeddingModel)
    }

    /// The part of a Gemma 4 file the loader does not run yet, read off its
    /// tensor table: the E2B/E4B's per-layer embeddings (FUTURE_WORK #115), or
    /// a scale on its experts' gate or up projection, which llama.cpp reads
    /// and no file met so far carries. Loaded without it, either would run
    /// with that part missing and write nonsense, so it is refused instead.
    /// The 26B-A4B's experts run (#115 (3b)).
    fn unbuilt_gemma4_part(&self) -> Option<&'static str> {
        if self.architecture != "gemma4" {
            return None;
        }
        if self.tensors.contains_key("per_layer_token_embd.weight") {
            return Some("per-layer embeddings (the E2B and E4B sizes)");
        }
        const UNREAD_SCALES: [&str; 3] = [
            ".ffn_gate_exps.scale",
            ".ffn_up_exps.scale",
            ".ffn_gate_up_exps.scale",
        ];
        self.tensors
            .keys()
            .any(|name| UNREAD_SCALES.iter().any(|s| name.ends_with(s)))
            .then_some("a scale on its experts' gate or up projection")
    }

    /// The weight bytes ONE token reads per layer, on average over the model:
    /// every `blk.*` tensor, a mixture-of-experts layer's expert stacks
    /// (`_exps`) only for the share a token is routed to (`expert_used_count /
    /// expert_count`), and the output projection, which every token reads whole
    /// (the token embedding is one row, so it counts only where it IS the
    /// output projection — a tied model). Decode reads these bytes once per token,
    /// so time per layer scales with them — the quantity the route planner's
    /// per-layer prior is a fraction of (FUTURE_WORK #250; Petals measures each
    /// model's own blocks for the same reason).
    ///
    /// `None` when the header lists no layers, or lists expert stacks but not
    /// how many experts a token uses: a guess at the active share could be off
    /// by the whole expert count.
    pub fn active_bytes_per_layer(&self) -> Option<f64> {
        self.bytes_per_layer(true)
    }

    /// The weight bytes per layer the model HOLDS — every expert of a
    /// mixture-of-experts layer, not the share a token is routed to — counted
    /// over the same tensors as [`Self::active_bytes_per_layer`]. Over it, how
    /// many times more a model holds than one token reads: 1 for a dense model,
    /// 8.5 for Qwen3-30B-A3B at Q2_K and 9.6 at Q4_K_M (8 of 128 experts; its
    /// attention and output projection are read whole). `None` where the active figure
    /// is, so the two are always read as a pair.
    pub fn stored_bytes_per_layer(&self) -> Option<f64> {
        self.active_bytes_per_layer()?;
        self.bytes_per_layer(false)
    }

    fn bytes_per_layer(&self, routed_share_only: bool) -> Option<f64> {
        if self.block_count == 0 {
            return None;
        }
        let tied = !self.tensors.contains_key("output.weight");
        let mut bytes = 0.0_f64;
        for (name, loc) in &self.tensors {
            let size = loc.size as f64;
            if name.starts_with("blk.") {
                if name.contains("_exps") && routed_share_only {
                    if self.expert_count == 0 || self.expert_used_count == 0 {
                        return None;
                    }
                    bytes += size * self.expert_used_count.min(self.expert_count) as f64
                        / self.expert_count as f64;
                } else {
                    bytes += size;
                }
            } else if name != "token_embd.weight" || tied {
                bytes += size;
            }
        }
        Some(bytes / self.block_count as f64)
    }
}

/// Tokenizer metadata extracted from GGUF header — consolidates all vocab/BOS/EOS/template
/// extraction that was previously duplicated across 9 call sites.
#[derive(Clone, Debug, Default)]
pub struct GgufTokenizerMeta {
    pub vocab: Vec<String>,
    pub bos_token_id: Option<u32>,
    pub eos_token_id: Option<u32>,
    /// All EOS token IDs (primary + extras from `eos_token_ids` array).
    pub eos_token_ids: Vec<u32>,
    pub chat_template: Option<String>,
    pub merges: Vec<String>,
    pub pre_tokenizer: String,
    pub tokenizer_model: String,
    pub scores: Vec<f32>,
    pub add_space_prefix: bool,
    pub add_bos_token: bool,
    /// `tokenizer.ggml.token_type`, one per vocabulary entry (GGUF's
    /// `llama_token_type`: 1 normal, 2 unknown, 3 CONTROL, 4 USER_DEFINED,
    /// 5 unused, 6 byte). Empty when the file carries none. **The vocabulary's
    /// own answer to "is this a special token"** — see
    /// `tokenizer::declared_special`.
    pub token_types: Vec<i32>,
    /// `general.architecture` ("llama" when absent, as [`gguf_arch_str`]
    /// answers). Read by [`special_token_spacing_for`] — a property of the
    /// family's tokenizer the GGUF has no key for.
    pub architecture: String,
}

/// Pre-tokenizers that make a GPT-2-style BPE vocabulary prepend BOS when the
/// GGUF does not say — llama.cpp's own list (`llama_vocab::impl::load`, the
/// `add_bos = true` arms of the pre-tokenizer switch). Every OTHER BPE vocabulary
/// defaults to no BOS there, which is what the model was converted against.
const BPE_PRE_TOKENIZERS_THAT_ADD_BOS: &[&str] = &[
    "llama3",
    "llama-v3",
    "llama-bpe",
    "falcon3",
    "falcon-h1",
    "pixtral",
    "midm-2.0",
    "lfm2",
    "jina-v5-nano",
    "tekken",
    "chameleon",
];

/// Pre-tokenizers whose models take NO BOS whatever the file declares:
/// llama.cpp sets `special_bos_id = LLAMA_TOKEN_NULL` for them. GLM-4's GGUF
/// declares `bos_token_id = <|endoftext|>` — which is also one of its EOS
/// tokens — and omits `add_bos_token`; prepending it put an end-of-text marker
/// in front of every prompt (FUTURE_WORK #97).
const BPE_PRE_TOKENIZERS_WITHOUT_BOS: &[&str] = &["glm4", "chatglm-bpe"];

/// Whether a prompt gets a BOS token, by llama.cpp's rules
/// (`llama_vocab::impl::load`): the file's own `add_bos_token` when present;
/// otherwise TRUE for a SentencePiece vocabulary (`tokenizer.ggml.model =
/// "llama"`) and, for a GPT-2-style BPE one, true only for the pre-tokenizers
/// that ask for it. A pre-tokenizer with no BOS at all overrides the key.
///
/// The old rule defaulted every vocabulary to true, on a comment saying the
/// flag "is consumed solely by the SPM path" — it was not: `SplitTokenizer`
/// prepends BOS for every variant.
///
/// Gemma 4's vocabulary takes BOS even where its file says not to: llama.cpp
/// overrides the key "for Gemma4" (llama.cpp PR #21500).
fn add_bos_by_llama_cpp_rules(declared: Option<bool>, tokenizer_model: &str, pre: &str) -> bool {
    if BPE_PRE_TOKENIZERS_WITHOUT_BOS.contains(&pre) {
        return false;
    }
    if tokenizer_model == "gemma4" {
        return true;
    }
    declared.unwrap_or(match tokenizer_model {
        "llama" => true,
        "gpt2" => BPE_PRE_TOKENIZERS_THAT_ADD_BOS.contains(&pre),
        _ => false,
    })
}

/// How a SentencePiece vocabulary of this ARCHITECTURE treats the text beside
/// a special token — the one place it is decided.
///
/// `phi3` is the one family that differs: its Hugging Face `tokenizer.json`
/// marks every added token but `<unk>`, `<s>` and `<|endoftext|>` `rstrip`,
/// and prefixes `▁` per segment. Checked against that file on this fleet's
/// Phi-3.5-mini: 5 of 8 prompts matched before, every chat prompt differing,
/// each by the newline after `<|user|>`/`<|end|>`/`<|assistant|>`. A `phi3`
/// GGUF with a GPT-2 vocabulary (Phi-4-mini) never reaches the SentencePiece
/// encoder, so this cannot touch it.
///
/// Keyed on the architecture, not on `general.name` as llama.cpp keys it
/// ("phi-3"/"phi3" in the lowercased name), which misses this fleet's
/// "Phi 3.5 Mini Instruct".
pub(crate) fn special_token_spacing_for(architecture: &str) -> SpecialTokenSpacing {
    match architecture {
        "phi3" => SpecialTokenSpacing::StripAfterSpecialAndPrefixEach,
        _ => SpecialTokenSpacing::FirstSegmentOnly,
    }
}

/// Token strings that END GENERATION, searched for BY NAME in the vocabulary.
///
/// **A model's declared EOS is not reliably the token it ends its turn with.**
/// Phi-3/3.5/4 declare `eos_token_id = <|endoftext|>` and close every assistant
/// turn with `<|end|>` — a different token, 32007 in Phi-3.5 — and their GGUFs
/// carry no `eot_token_id` key to say so. Nothing then stopped generation: the
/// model ended its turn and carried on, inventing a second assistant turn and a
/// fabricated user turn until `max_tokens`. Reproduced on v0.3.171 with a plain
/// no-tools request to Phi-3.5 ("Say exactly: hello" → 120/120 tokens,
/// `finish_reason: "length"`, a fabricated Chinese user turn at the end).
///
/// The symptom differs by vocabulary family, from this one cause, which is why
/// it read as two unrelated bugs in the field:
///
/// - **SentencePiece vocab** (Phi-3.5's GGUF): `PieceDecoder::decode` maps a
///   `<…>` token the vocabulary does not declare text to empty, so the CONTROL
///   `<|end|>` vanishes and the run-on is INVISIBLE.
/// - **GPT-2 byte BPE vocab** (Phi-4-mini's GGUF): the same token decodes to
///   its literal characters, so `<|end|>` LEAKS into visible content and the
///   reply runs on after it.
///
/// Searching the vocabulary by name is llama.cpp's approach
/// (`llama_vocab::impl::load`, which populates `special_eog_ids` from this same
/// kind of candidate list) rather than a per-architecture table of ids: a name
/// is stable across quantisations and vocab sizes, an id is not.
///
/// Kept to markers whose ONLY role is ending a turn. Three deliberate absences,
/// each one a way this could go wrong:
///
/// - **`<|user|>` / `<|assistant|>` / `<|system|>`** open a turn. A model
///   emitting one has gone wrong, but that is a stop-string concern
///   (`chat_template::extract_stop_strings`), not an end-of-generation token.
/// - **`</s>` and `<eos>`**, which llama.cpp's own candidate list does carry.
///   Every marker above is a `<|…|>` form whose only role in any family is
///   ending a turn; these two instead sit in the vocabulary of families that
///   never emit them — Phi-3.5's SPM vocab holds `</s>` at id 2 and ends its
///   turns with `<|end|>`. Adopting them would add a silent-truncation risk to
///   every such model with nothing demonstrating it is needed, and the severity
///   ordering below settles that: a wrong EOS truncates SILENTLY, an unknown one
///   at worst runs to `max_tokens`. A genuine Llama-2 vocabulary still gets id 2
///   from the narrowly-scoped `ids.is_empty() && plausible(2)` branch in
///   [`GgufTokenizerMeta::eos_tokens_with_arch_fallback`], which fires only when
///   the model declared nothing at all.
const END_OF_GENERATION_TOKENS: &[&str] = &[
    "<|end|>",
    "<|eot_id|>",
    "<|eom_id|>",
    "<|im_end|>",
    "<|endoftext|>",
    "<|end_of_text|>",
    "<end_of_turn>",
    "<|return|>",
    "<|call|>",
    // Gemma 4 closes a turn with `<turn|>` and hands a tool call back with
    // `<|tool_response>` — both on llama.cpp's list, both found only there.
    "<turn|>",
    "<|tool_response>",
];

/// The token whose end-of-generation meaning is CONDITIONAL on its neighbours.
const CONDITIONAL_END_TOKEN: &str = "<|end|>";

/// A vocabulary carrying any of these uses `<|end|>` to close a MESSAGE, not to
/// end generation — so `<|end|>` must NOT stop the reply there.
///
/// `<|return|>` + `<|call|>` is OpenAI's harmony format (gpt-oss); `<|calls|>` +
/// `<|flush|>` is solar-open. Both emit `<|end|>` between messages of a reply
/// that is still being written, so stopping on it truncates every such reply at
/// its first message. **This guard is the reason to read upstream before
/// shipping a one-line fix**: adding `<|end|>` unconditionally is correct for
/// Phi and silently breaks harmony models, and llama.cpp carries the same
/// exclusion for the same reason.
const END_IS_MESSAGE_BOUNDARY_MARKERS: &[&str] =
    &["<|return|>", "<|call|>", "<|calls|>", "<|flush|>"];

/// Does this vocabulary entry look like an end-of-turn marker rather than an
/// ordinary token?
///
/// Every family wraps its turn markers in angle brackets — `</s>`, `<|eot_id|>`,
/// `<|im_end|>`, `<end_of_turn>` — while ordinary tokens (`#`, `$`, `the`) are
/// bare. Byte tokens are wrapped too (`<0x0A>`) and are explicitly NOT turn
/// markers, which is the one case a naive bracket check gets wrong.
///
/// This exists so a fallback EOS id can be VERIFIED instead of assumed. See
/// [`GgufTokenizerMeta::eos_tokens_with_arch_fallback`] for what assuming cost.
pub fn looks_like_end_of_turn(token: &str) -> bool {
    let t = token.trim();
    if t.len() <= 2 || !t.starts_with('<') || !t.ends_with('>') {
        return false;
    }
    let inner = &t[1..t.len() - 1];
    let inner = inner.strip_prefix('|').unwrap_or(inner);
    let inner = inner.strip_suffix('|').unwrap_or(inner);
    !inner.to_ascii_lowercase().starts_with("0x")
}

impl GgufTokenizerMeta {
    /// Extract tokenizer metadata from a GGUF header file on disk.
    pub fn from_gguf_file(path: &Path) -> Result<Self, SwarmError> {
        let ct = read_gguf_header(path)?;
        Ok(Self::from_content(&ct))
    }

    /// Extract tokenizer metadata from an already-parsed GGUF Content.
    pub fn from_content(ct: &gguf_file::Content) -> Self {
        let md = &ct.metadata;

        let vocab: Vec<String> = md
            .get("tokenizer.ggml.tokens")
            .and_then(|v| v.to_vec().ok())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.to_string().ok().cloned())
                    .collect()
            })
            .unwrap_or_default();

        let bos_token_id = md
            .get("tokenizer.ggml.bos_token_id")
            .and_then(|v| v.to_u32().ok());

        let eos_token_id = md
            .get("tokenizer.ggml.eos_token_id")
            .and_then(|v| v.to_u32().ok());

        // Collect all EOS IDs: primary + extras array
        let mut eos_ids = Vec::new();
        if let Some(id) = eos_token_id {
            eos_ids.push(id);
        }
        if let Some(extra) = md
            .get("tokenizer.ggml.eos_token_ids")
            .and_then(|v| v.to_vec().ok())
        {
            for v in extra {
                if let Ok(id) = v.to_u32() {
                    if !eos_ids.contains(&id) {
                        eos_ids.push(id);
                    }
                }
            }
        }

        let chat_template = md
            .get("tokenizer.chat_template")
            .and_then(|v| v.to_string().ok().cloned());

        let merges: Vec<String> = md
            .get("tokenizer.ggml.merges")
            .and_then(|v| v.to_vec().ok())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.to_string().ok().cloned())
                    .collect()
            })
            .unwrap_or_default();

        let pre_tokenizer = md
            .get("tokenizer.ggml.pre")
            .and_then(|v| v.to_string().ok().cloned())
            .unwrap_or_else(|| "gpt2".to_string());

        let tokenizer_model = md
            .get("tokenizer.ggml.model")
            .and_then(|v| v.to_string().ok().cloned())
            .unwrap_or_else(|| "gpt2".to_string());

        let scores: Vec<f32> = md
            .get("tokenizer.ggml.scores")
            .and_then(|v| v.to_vec().ok())
            .map(|arr| arr.iter().filter_map(|v| v.to_f32().ok()).collect())
            .unwrap_or_default();

        let add_space_prefix = md
            .get("tokenizer.ggml.add_space_prefix")
            .and_then(|v| v.to_bool().ok())
            .unwrap_or(true);

        // Defaulting to false meant every Llama-family GGUF that simply omits
        // the key — TinyLlama and Phi-3.5 among them — was prefilled with no
        // BOS at position 0, which is out-of-distribution for models trained
        // with one and produced degenerate replies. Defaulting to TRUE for
        // every vocabulary then gave GLM-4 an end-of-text marker in front of
        // every prompt. llama.cpp's per-tokenizer rule is the reference.
        let add_bos_token = add_bos_by_llama_cpp_rules(
            md.get("tokenizer.ggml.add_bos_token")
                .and_then(|v| v.to_bool().ok()),
            &tokenizer_model,
            &pre_tokenizer,
        );

        let token_types: Vec<i32> = md
            .get("tokenizer.ggml.token_type")
            .and_then(|v| v.to_vec().ok())
            .map(|arr| arr.iter().filter_map(|v| v.to_i32().ok()).collect())
            .unwrap_or_default();

        Self {
            vocab,
            bos_token_id,
            eos_token_id,
            eos_token_ids: eos_ids,
            chat_template,
            merges,
            pre_tokenizer,
            tokenizer_model,
            scores,
            add_space_prefix,
            add_bos_token,
            token_types,
            architecture: gguf_arch_str(ct),
        }
    }

    /// Resolve BOS token ID to its string representation from the vocab.
    pub fn bos_string(&self) -> String {
        self.bos_token_id
            .and_then(|id| self.vocab.get(id as usize))
            .cloned()
            .unwrap_or_default()
    }

    /// Resolve primary EOS token ID to its string representation from the vocab.
    pub fn eos_string(&self) -> String {
        self.eos_token_id
            .and_then(|id| self.vocab.get(id as usize))
            .cloned()
            .unwrap_or_default()
    }

    /// End-of-generation ids found by NAME in this model's own vocabulary.
    ///
    /// The answer to "which tokens actually end a reply", independent of what
    /// the GGUF declared as EOS — see [`END_OF_GENERATION_TOKENS`] for the
    /// Phi-3/3.5/4 case that made this necessary and
    /// [`END_IS_MESSAGE_BOUNDARY_MARKERS`] for the harmony exclusion.
    ///
    /// Needs only the vocabulary, so every path that resolves EOS ids can share
    /// it whether or not it knows the architecture.
    pub fn end_of_generation_ids_from_vocab(&self) -> Vec<u32> {
        if self.vocab.is_empty() {
            return Vec::new();
        }
        // `<|end|>` ends generation only when this vocabulary is not one of the
        // formats that use it as a message separator.
        let end_is_terminal = !self
            .vocab
            .iter()
            .any(|t| END_IS_MESSAGE_BOUNDARY_MARKERS.contains(&t.as_str()));

        self.vocab
            .iter()
            .enumerate()
            .filter(|(_, tok)| {
                let t = tok.as_str();
                END_OF_GENERATION_TOKENS.contains(&t)
                    && (end_is_terminal || t != CONDITIONAL_END_TOKEN)
            })
            .map(|(id, _)| id as u32)
            .collect()
    }

    /// Get EOS token IDs with architecture-specific fallbacks.
    ///
    /// **Every candidate this adds is checked against the vocabulary first.**
    /// The old `ids.push(2)` — commented "common default" — is Llama-2's `</s>`
    /// and an ORDINARY TOKEN in every later family: `#` in Qwen2.5, an
    /// unremarkable byte token in Llama-3 and Gemma. Treating it as end-of-turn
    /// truncates a reply at its first `#`, which for a coding model is a Rust
    /// attribute, a Python comment or a markdown heading — i.e. the first token
    /// of a perfectly good answer. The caller sees `completion_tokens: 1` and
    /// `finish_reason: "stop"` with no error and no stop sequence matched.
    ///
    /// A wrong EOS truncates SILENTLY; an unknown one at worst lets the reply
    /// run to `max_tokens`. Those are not close in severity, so when nothing can
    /// be verified this returns EMPTY rather than guessing.
    ///
    /// **A declared EOS is trusted but never assumed complete.** The per-family
    /// id lists below only ever ran when a GGUF declared NOTHING, so a model
    /// that declares one EOS and ends its turns with another got no help from
    /// them at all — which is every Phi-3/3.5/4 GGUF. The vocabulary search in
    /// [`Self::end_of_generation_ids_from_vocab`] is merged in unconditionally
    /// for that reason.
    pub fn eos_tokens_with_arch_fallback(&self, arch: &str) -> Vec<u32> {
        let mut ids = self.eos_token_ids.clone();
        // A declared EOS is trusted, but it is not assumed to be COMPLETE: a
        // model that ends its turns with a token it did not declare stops on
        // nothing. Merged before the `ids.is_empty()` fallback below so a
        // vocabulary that names its own turn-ender is never given a guess
        // instead.
        for id in self.end_of_generation_ids_from_vocab() {
            if !ids.contains(&id) {
                ids.push(id);
            }
        }
        // Verify against the vocabulary when we have one; trust the
        // architecture when we do not, since an unreadable vocab is not
        // evidence against an id the architecture is sure of.
        let plausible = |id: u32| {
            self.vocab.is_empty()
                || self
                    .vocab
                    .get(id as usize)
                    .is_some_and(|t| looks_like_end_of_turn(t))
        };
        if ids.is_empty() && plausible(2) {
            ids.push(2); // Llama-2's `</s>`, and only when the vocab agrees
        }
        // Qwen2 uses additional EOS tokens
        if arch.starts_with("qwen") {
            for &extra in &[151643u32, 151645] {
                if !ids.contains(&extra) && plausible(extra) {
                    ids.push(extra);
                }
            }
        }
        // Gemma uses token 107 (<end_of_turn>) as EOS
        if (arch == "gemma" || arch == "gemma2") && !ids.contains(&107) && plausible(107) {
            ids.push(107);
        }
        // Gemma 4 declares `<turn|>` and also ends a plain completion with
        // `<eos>` (id 1), which llama.cpp stops on: `<eos>` is on its list "for
        // gemma4" (`llama_vocab::impl::load`). Kept to this family — see
        // `END_OF_GENERATION_TOKENS` on why `<eos>` is not on that list.
        if arch == "gemma4" && !ids.contains(&1) && plausible(1) {
            ids.push(1);
        }
        ids
    }

    /// Build a `SplitTokenizer` from extracted metadata, or `None` if vocab is empty.
    pub fn build_tokenizer(&self) -> Option<SplitTokenizer> {
        if self.vocab.is_empty() {
            return None;
        }
        if !self.merges.is_empty() {
            Some(SplitTokenizer::from_bpe(
                &self.vocab,
                &self.merges,
                &self.pre_tokenizer,
                &self.tokenizer_model,
                self.add_bos_token,
                self.bos_token_id,
                &self.token_types,
            ))
        } else if self.tokenizer_model == "llama" && !self.scores.is_empty() {
            Some(
                SplitTokenizer::from_sentencepiece(
                    &self.vocab,
                    &self.scores,
                    self.add_space_prefix,
                    self.add_bos_token,
                    self.bos_token_id,
                    &self.token_types,
                )
                .with_special_token_spacing(special_token_spacing_for(&self.architecture)),
            )
        } else {
            None
        }
    }
}

// ── GGUF Header Extraction ──

/// Save the raw GGUF header (metadata + tensor info table) to a file.
/// The header is everything from byte 0 up to (but not including) `tensor_data_offset`.
/// This allows nodes without shard_000 to reconstruct the GGUF parsing context.
///
/// The source can be a full GGUF file, OR shard_000.bin (which is the first
/// 512MB of the GGUF and always contains the complete header, since headers
/// are typically only a few MB).
pub fn save_gguf_header(gguf_or_shard0_path: &Path, output_path: &Path) -> Result<(), SwarmError> {
    let ct = read_gguf_header(gguf_or_shard0_path)?;

    let header_size = ct.tensor_data_offset as usize;
    // SEC: Cap header allocation to prevent OOM from malicious GGUF files
    const MAX_GGUF_HEADER_SIZE: usize = 64 * 1024 * 1024; // 64 MB
    if header_size > MAX_GGUF_HEADER_SIZE {
        return Err(SwarmError::Internal(format!(
            "GGUF header too large: {} bytes (max {})",
            header_size, MAX_GGUF_HEADER_SIZE
        )));
    }
    let mut header_buf = vec![0u8; header_size];
    // A second, unbuffered handle on purpose: this is one `read_exact` of the
    // whole header, not the parse's thousands of small reads, so a buffer would
    // only copy the bytes twice. Freshly opened, so it is already at offset 0.
    let mut file = std::fs::File::open(gguf_or_shard0_path).map_err(SwarmError::Io)?;
    file.read_exact(&mut header_buf).map_err(SwarmError::Io)?;

    if let Some(parent) = output_path.parent() {
        std::fs::create_dir_all(parent).map_err(SwarmError::Io)?;
    }
    // SEC: Atomic write to prevent corruption on kill/crash
    let tmp_path = output_path.with_extension("bin.tmp");
    std::fs::write(&tmp_path, &header_buf).map_err(SwarmError::Io)?;
    std::fs::rename(&tmp_path, output_path).map_err(SwarmError::Io)?;

    tracing::info!(
        header_bytes = header_size,
        path = %output_path.display(),
        "Saved GGUF header for shard-only operation"
    );
    Ok(())
}

/// Try to extract the GGUF header from shard_000.bin if it exists in the model directory.
/// This enables shard-only operation without needing the full GGUF or a `source_path`.
pub fn ensure_gguf_header(model_dir: &Path) -> Result<(), SwarmError> {
    let header_path = model_dir.join(crate::model::shard::HEADER_FILENAME);
    if header_path.exists() {
        return Ok(());
    }

    // shard_000.bin contains the GGUF header (first ~6MB of the file)
    let shard0_path = model_dir.join("shard_000.bin");
    if shard0_path.exists() {
        tracing::info!(
            model_dir = %model_dir.display(),
            "Extracting GGUF header from shard_000.bin"
        );
        return save_gguf_header(&shard0_path, &header_path);
    }

    // Try source_path as a fallback (with path containment check)
    let source_path_file = model_dir.join("source_path");
    if source_path_file.exists() {
        if let Ok(path_str) = std::fs::read_to_string(&source_path_file) {
            let gguf_path = std::path::PathBuf::from(path_str.trim());
            // SEC: Canonicalize both paths to prevent traversal bypass.
            // If either fails, skip — don't fall back to raw uncanonicalized paths.
            let canonical = match gguf_path.canonicalize() {
                Ok(c) => c,
                Err(_) => {
                    tracing::warn!(path = %gguf_path.display(), "source_path canonicalize failed — skipping");
                    return Err(SwarmError::Internal("source_path not resolvable".into()));
                }
            };
            // Allow source_path to be anywhere (it's typically the original GGUF
            // outside the model dir). Just verify the path exists and is a file.
            if canonical.exists() && canonical.is_file() {
                tracing::info!(
                    gguf = %canonical.display(),
                    "Extracting GGUF header from source path"
                );
                return save_gguf_header(&canonical, &header_path);
            }
        }
    }

    Err(SwarmError::Internal(format!(
        "Cannot create gguf_header.bin: no shard_000.bin or source GGUF found in {}",
        model_dir.display()
    )))
}

#[cfg(test)]
mod gguf_parse_error_tests {
    use super::explain_gguf_parse_error;

    /// An `IQ*` file is named for what it is and what to pick instead; the
    /// number alone told a reader nothing.
    #[test]
    fn an_unreadable_quantization_is_named_with_the_way_out() {
        let e = candle_core::Error::Msg("unknown dtype for tensor 18".into());
        let msg = explain_gguf_parse_error(&e);
        assert!(msg.contains("IQ3_XXS") && msg.contains("Q4_K_M"), "{msg}");
    }

    /// Any other failure is left exactly as candle said it.
    #[test]
    fn other_parse_failures_pass_through() {
        let e = candle_core::Error::Msg("invalid magic 0x0".into());
        assert_eq!(explain_gguf_parse_error(&e), "invalid magic 0x0");
    }
}

#[cfg(test)]
mod add_bos_rule_tests {
    use super::add_bos_by_llama_cpp_rules as rule;

    /// Phi-3 is the one SentencePiece family whose turn markers strip the
    /// whitespace after them, and it is recognised by ARCHITECTURE — the file on
    /// this fleet is named "Phi 3.5 Mini Instruct", which llama.cpp's name match
    /// misses. Every other architecture keeps what the encoder always did.
    #[test]
    fn phi3_spacing_is_chosen_by_architecture() {
        use super::special_token_spacing_for as spacing;
        use crate::inference::tokenizer::SpecialTokenSpacing::*;
        assert_eq!(spacing("phi3"), StripAfterSpecialAndPrefixEach);
        for arch in ["llama", "gemma2", "qwen2", "glm4", "mistral", ""] {
            assert_eq!(spacing(arch), FirstSegmentOnly, "{arch}");
        }
    }

    /// llama.cpp's rules (`llama_vocab::impl::load`), one arm each — the
    /// reference the old blanket `unwrap_or(true)` disagreed with (#97).
    #[test]
    fn bos_follows_llama_cpp_when_the_file_does_not_say() {
        // GLM-4 declares <|endoftext|> as BOS and omits the key: no BOS.
        assert!(!rule(None, "gpt2", "glm4"), "glm4 takes no BOS");
        // ...and none even if a converter wrote the key, as llama.cpp nulls it.
        assert!(!rule(Some(true), "gpt2", "chatglm-bpe"));
        // Llama-3 omits the key too, and DOES take one: the pre-tokenizer says so.
        assert!(rule(None, "gpt2", "llama-bpe"));
        // Any other BPE vocabulary defaults to none.
        assert!(!rule(None, "gpt2", "qwen2"));
        assert!(!rule(None, "gpt2", "deepseek-llm"));
        // SentencePiece defaults to one — TinyLlama and Phi-3.5 depend on it.
        assert!(rule(None, "llama", "default"));
        // A key that is present is the answer.
        assert!(!rule(Some(false), "llama", "default"));
        assert!(rule(Some(true), "gpt2", "qwen2"));
        // ...except Gemma 4's, which llama.cpp gives a BOS whatever it says.
        assert!(rule(Some(false), "gemma4", ""));
        assert!(rule(None, "gemma4", ""));
    }
}

#[cfg(test)]
mod eos_fallback_tests {
    use super::*;

    fn meta(vocab: Vec<&str>, declared: Vec<u32>) -> GgufTokenizerMeta {
        GgufTokenizerMeta {
            vocab: vocab.into_iter().map(String::from).collect(),
            eos_token_ids: declared,
            ..Default::default()
        }
    }

    /// A vocabulary where id 2 is an ordinary character — the shape of every
    /// modern family. Qwen2.5-Coder's real vocabulary starts `!`, `"`, `#`.
    fn qwen_like() -> Vec<&'static str> {
        let mut v = vec!["!", "\"", "#", "$", "%"];
        v.resize(151_646, "tok");
        v[151_643] = "<|endoftext|>";
        v[151_645] = "<|im_end|>";
        v
    }

    /// The reported failure, in miniature. A model whose GGUF declares no EOS
    /// must NOT be given Llama-2's id 2 when its own vocabulary says that id is
    /// `#` — a coding reply opening with `#[derive(...)]` or a markdown heading
    /// was cut to one token and reported as `finish_reason: "stop"`.
    #[test]
    fn an_ordinary_token_is_never_adopted_as_end_of_turn() {
        let ids = meta(qwen_like(), vec![]).eos_tokens_with_arch_fallback("qwen2");
        assert!(
            !ids.contains(&2),
            "id 2 is '#' in this vocabulary and must not end a turn: {ids:?}"
        );
        assert!(ids.contains(&151_645), "the real <|im_end|> must be there");
    }

    /// The case the old default was actually written for still works: a
    /// Llama-2-era vocabulary really does end its turn on id 2, and its own
    /// vocabulary says so.
    #[test]
    fn a_genuine_llama2_end_of_turn_is_still_adopted() {
        let mut v = vec!["<unk>", "<s>", "</s>", "a", "b"];
        v.resize(32_000, "tok");
        let ids = meta(v, vec![]).eos_tokens_with_arch_fallback("llama");
        assert_eq!(ids, vec![2], "'</s>' is a real turn marker: {ids:?}");
    }

    /// What the model declares always wins — this must not start second-guessing
    /// a GGUF that states its own EOS.
    #[test]
    fn a_declared_end_of_turn_is_left_alone() {
        let ids = meta(qwen_like(), vec![151_645]).eos_tokens_with_arch_fallback("qwen2");
        assert!(ids.contains(&151_645));
        assert!(!ids.contains(&2));
    }

    /// An unreadable vocabulary is not evidence against an id the architecture
    /// is sure of — otherwise a model with no vocab loses its EOS entirely.
    #[test]
    fn an_empty_vocabulary_still_trusts_the_architecture() {
        let ids = meta(vec![], vec![]).eos_tokens_with_arch_fallback("qwen2");
        assert!(ids.contains(&151_643) && ids.contains(&151_645));
    }

    /// A vocabulary shaped like Phi-3/3.5: it DECLARES `<|endoftext|>` as EOS
    /// and ends every turn with `<|end|>`, a different token.
    fn phi_like() -> Vec<&'static str> {
        let mut v = vec!["<unk>", "<s>", "</s>", "a", "b"];
        v.resize(32_011, "tok");
        v[32_000] = "<|endoftext|>";
        v[32_001] = "<|assistant|>";
        v[32_006] = "<|system|>";
        v[32_007] = "<|end|>";
        v[32_010] = "<|user|>";
        v
    }

    /// The field failure, in miniature. Phi declares one EOS and ends its turns
    /// with another, and the per-family lists only ever ran when a GGUF declared
    /// NOTHING — so the declared `<|endoftext|>` suppressed them and `<|end|>`
    /// stopped nothing. The model ended its turn and carried on inventing
    /// further turns to `max_tokens`.
    #[test]
    fn a_turn_ender_the_model_did_not_declare_is_still_found() {
        let ids = meta(phi_like(), vec![32_000]).eos_tokens_with_arch_fallback("phi3");
        assert!(
            ids.contains(&32_007),
            "<|end|> is how phi3 ends every turn and must stop generation: {ids:?}"
        );
        assert!(
            ids.contains(&32_000),
            "the declared EOS must survive: {ids:?}"
        );
    }

    /// The absence that keeps this fix from trading one silent failure for
    /// another. Phi-3.5's SPM vocabulary carries `</s>` at id 2 and never emits
    /// it — it ends turns with `<|end|>` — so adopting `</s>` would truncate
    /// silently, which this file documents as much worse than running on.
    #[test]
    fn a_marker_the_family_never_emits_is_not_adopted() {
        let ids = meta(phi_like(), vec![32_000]).eos_tokens_with_arch_fallback("phi3");
        assert!(
            !ids.contains(&2),
            "phi ends turns with <|end|>, not </s>: {ids:?}"
        );
    }

    /// Opening a turn is not ending one. `<|user|>` and `<|assistant|>` mean the
    /// model has gone wrong and belong in the stop STRINGS, but adopting them as
    /// end-of-generation tokens would end a reply on a token the model may
    /// legitimately be unable to avoid in some templates.
    #[test]
    fn a_turn_opener_is_not_adopted_as_end_of_generation() {
        let ids = meta(phi_like(), vec![32_000]).eos_tokens_with_arch_fallback("phi3");
        assert!(!ids.contains(&32_010), "<|user|> opens a turn: {ids:?}");
        assert!(
            !ids.contains(&32_001),
            "<|assistant|> opens a turn: {ids:?}"
        );
    }

    /// The trap that makes the one-line version of this fix wrong. In OpenAI's
    /// harmony format (gpt-oss) `<|end|>` separates messages INSIDE one reply,
    /// so stopping there truncates every such reply at its first message.
    /// llama.cpp carries the same exclusion, keyed on the same neighbours.
    #[test]
    fn end_does_not_stop_generation_where_it_separates_messages() {
        let mut v = vec!["!", "\"", "#"];
        v.resize(201_000, "tok");
        v[200_002] = "<|end|>";
        v[200_007] = "<|return|>";
        v[200_008] = "<|call|>";
        v[200_009] = "<|endoftext|>";
        let ids = meta(v, vec![200_009]).eos_tokens_with_arch_fallback("gpt-oss");
        assert!(
            !ids.contains(&200_002),
            "<|end|> only separates harmony messages and must not end the reply: {ids:?}"
        );
        assert!(
            ids.contains(&200_007) && ids.contains(&200_008),
            "harmony ends a reply on <|return|> / <|call|>: {ids:?}"
        );
    }

    /// An empty vocabulary yields no name matches, and must not change what the
    /// architecture fallbacks already did.
    #[test]
    fn the_vocabulary_search_is_silent_without_a_vocabulary() {
        assert!(meta(vec![], vec![])
            .end_of_generation_ids_from_vocab()
            .is_empty());
        let ids = meta(vec![], vec![]).eos_tokens_with_arch_fallback("qwen2");
        assert!(ids.contains(&151_643) && ids.contains(&151_645));
    }

    #[test]
    fn turn_markers_are_told_apart_from_ordinary_and_byte_tokens() {
        for marker in ["</s>", "<|eot_id|>", "<|im_end|>", "<end_of_turn>", "<eos>"] {
            assert!(looks_like_end_of_turn(marker), "{marker} is a turn marker");
        }
        // Byte tokens are bracketed but are NOT turn markers — the one case a
        // naive bracket check gets wrong.
        for ordinary in ["#", "$", "the", "<0x0A>", "<0xFF>", "<>", ""] {
            assert!(
                !looks_like_end_of_turn(ordinary),
                "{ordinary:?} must not be treated as a turn marker"
            );
        }
    }
}

#[cfg(test)]
mod kv_layer_tests {
    use super::*;

    /// A header whose tensor table names `names`.
    fn header(names: &[String]) -> gguf_file::Content {
        gguf_file::Content {
            magic: gguf_file::VersionedMagic::GgufV3,
            metadata: HashMap::new(),
            tensor_infos: names
                .iter()
                .map(|n| {
                    (
                        n.clone(),
                        gguf_file::TensorInfo {
                            ggml_dtype: candle_core::quantized::GgmlDType::F32,
                            shape: candle_core::Shape::from(1usize),
                            offset: 0,
                        },
                    )
                })
                .collect(),
            tensor_data_offset: 0,
        }
    }

    /// Qwen 3.5's shape: a full-attention layer every fourth (llama.cpp:
    /// recurrent where `(il + 1) % 4 != 0`), told apart as the loader tells
    /// them — by `ssm_alpha`.
    #[test]
    fn only_a_models_attention_layers_keep_a_kv_cache() {
        let names: Vec<String> = (0..8)
            .map(|il| {
                if (il + 1) % 4 != 0 {
                    format!("blk.{il}.ssm_alpha.weight")
                } else {
                    format!("blk.{il}.attn_k.weight")
                }
            })
            .collect();
        let ct = header(&names);
        assert!(layer_is_recurrent(&ct, 0));
        assert!(!layer_is_recurrent(&ct, 3));
        assert_eq!(kv_layers(&ct, 0, 8).collect::<Vec<_>>(), [3, 7]);
        assert_eq!(kv_layers(&ct, 0, 4).count(), 1);
        assert_eq!(kv_layers(&ct, 4, 7).count(), 0);
    }

    /// A transformer: every layer keeps one.
    #[test]
    fn every_layer_of_a_transformer_keeps_a_kv_cache() {
        let names: Vec<String> = (0..6).map(|il| format!("blk.{il}.attn_k.weight")).collect();
        assert_eq!(kv_layers(&header(&names), 0, 6).count(), 6);
        assert_eq!(kv_layers(&header(&names), 2, 5).count(), 3);
    }
}

#[cfg(test)]
mod active_bytes_tests {
    use super::*;

    /// A header of `layers` layers whose tensors have the given sizes.
    fn meta(layers: usize, experts: (usize, usize), tensors: &[(&str, u64)]) -> GgufTensorMeta {
        GgufTensorMeta {
            tensors: tensors
                .iter()
                .map(|(n, size)| {
                    (
                        n.to_string(),
                        TensorLocation {
                            offset: 0,
                            size: *size,
                        },
                    )
                })
                .collect(),
            tensor_data_offset: 0,
            model_name: None,
            head_count: 1,
            head_count_kv: 1,
            block_count: layers,
            embedding_length: 1,
            head_dim: 1,
            rope_dim: 1,
            rope_freq_base: 1.0,
            rms_norm_eps: 1e-6,
            expert_count: experts.0,
            expert_used_count: experts.1,
            architecture: "qwen3moe".into(),
            context_length: 0,
            pooling_type: 0,
            kv_layout: None,
        }
    }

    /// FUTURE_WORK #250: a token reads 8 of a layer's 128 experts, so the
    /// expert stacks count for 1/16 of their bytes; attention counts whole; the
    /// output projection is read every token and the embedding is one row.
    #[test]
    fn a_token_reads_the_experts_it_is_routed_to_and_the_layers_whole() {
        let m = meta(
            2,
            (128, 8),
            &[
                ("token_embd.weight", 1_000),
                ("output.weight", 200),
                ("blk.0.attn_q.weight", 100),
                ("blk.0.ffn_up_exps.weight", 1_600),
                ("blk.1.attn_q.weight", 100),
                ("blk.1.ffn_up_exps.weight", 1_600),
            ],
        );
        // (100 + 1600/16) × 2 layers + 200 output = 600, over 2 layers.
        assert_eq!(m.active_bytes_per_layer(), Some(300.0));
        // Held, every expert counts: (100 + 1600) × 2 + 200 = 3600, over 2.
        assert_eq!(m.stored_bytes_per_layer(), Some(1_800.0));
    }

    /// A dense model holds exactly what a token reads.
    #[test]
    fn a_dense_model_holds_what_a_token_reads() {
        let m = meta(
            1,
            (0, 0),
            &[("output.weight", 50), ("blk.0.attn_q.weight", 100)],
        );
        assert_eq!(m.stored_bytes_per_layer(), m.active_bytes_per_layer());
        assert_eq!(m.stored_bytes_per_layer(), Some(150.0));
    }

    /// A tied model reads its embedding as the output projection every token.
    #[test]
    fn a_tied_embedding_is_read_whole_as_the_output() {
        let m = meta(
            1,
            (0, 0),
            &[("token_embd.weight", 500), ("blk.0.attn_q.weight", 100)],
        );
        assert_eq!(m.active_bytes_per_layer(), Some(600.0));
    }

    /// Experts listed with no count of how many a token uses: unknown, never a
    /// guess that could be off by the whole expert count.
    #[test]
    fn experts_without_a_used_count_are_unknown() {
        let m = meta(
            1,
            (128, 0),
            &[("output.weight", 1), ("blk.0.ffn_up_exps.weight", 1_600)],
        );
        assert_eq!(m.active_bytes_per_layer(), None);
        assert_eq!(m.stored_bytes_per_layer(), None, "read as a pair");
    }
}

#[cfg(test)]
mod refusal_tests {
    use super::*;
    use candle_core::quantized::gguf_file::Value;

    /// A header with the keys `from_content` requires, for `arch`, plus
    /// `<arch>.pooling_type` when given — written and parsed in memory.
    fn header(arch: &str, pooling_type: Option<u32>) -> GgufTensorMeta {
        let key = |k: &str| format!("{arch}.{k}");
        let mut md: Vec<(String, Value)> = vec![
            ("general.architecture".into(), Value::String(arch.into())),
            (key("attention.head_count"), Value::U32(16)),
            (key("attention.head_count_kv"), Value::U32(8)),
            (key("block_count"), Value::U32(28)),
            (key("embedding_length"), Value::U32(1024)),
            (key("attention.layer_norm_rms_epsilon"), Value::F32(1e-6)),
        ];
        if let Some(p) = pooling_type {
            md.push((key("pooling_type"), Value::U32(p)));
        }
        let refs: Vec<(&str, &Value)> = md.iter().map(|(k, v)| (k.as_str(), v)).collect();
        let mut buf = std::io::Cursor::new(Vec::new());
        gguf_file::write(&mut buf, &refs, &[]).unwrap();
        buf.set_position(0);
        GgufTensorMeta::from_content(&gguf_file::Content::read(&mut buf).unwrap()).unwrap()
    }

    /// Field report 2026-10-15: Qwen3-Embedding-0.6B's header is a `qwen3`
    /// with `qwen3.pooling_type = 3` (LAST) — read off the real file. Ollama
    /// classes any GGUF naming a pooling type as an embedding model; NONE (0)
    /// or no key is a model that writes replies.
    #[test]
    fn a_header_that_names_a_pooling_type_is_an_embedding_model() {
        let embedding = header("qwen3", Some(3));
        assert_eq!(embedding.pooling_type, 3);
        assert_eq!(embedding.refusal(), Some(ModelRefusal::EmbeddingModel));
        assert!(matches!(
            ModelRefusal::EmbeddingModel.into_error(),
            SwarmError::Validation(m) if m.contains("embedding model")
        ));

        assert_eq!(header("qwen3", None).refusal(), None, "a chat model");
        assert_eq!(header("qwen3", Some(0)).refusal(), None, "NONE is not said");
        assert_eq!(
            header("starcoder2", None).refusal(),
            Some(ModelRefusal::Architecture("starcoder2".into())),
            "a family this build has no loader for is still refused for that"
        );
    }

    /// A Gemma 4 size whose layers this build does not run yet is refused by
    /// name, read off its tensor table — loaded as the dense layout it would
    /// run with its experts or per-layer embeddings missing.
    #[test]
    fn gemma4_sizes_this_build_cannot_run_are_refused() {
        let with = |tensor: &str| {
            let mut meta = header("gemma4", None);
            meta.tensors
                .insert(tensor.into(), TensorLocation { offset: 0, size: 1 });
            meta.refusal()
        };
        assert_eq!(header("gemma4", None).refusal(), None, "12B / 31B");
        assert_eq!(with("blk.0.attn_q.weight"), None);
        let refused = |r: Option<ModelRefusal>, part: &str| matches!(r, Some(ModelRefusal::Architecture(a)) if a.starts_with("gemma4 with") && a.contains(part));
        // The 26B-A4B's experts run (#115 (3b)); a scale on their gate or up
        // projection is not read, so a file carrying one is refused.
        assert_eq!(with("blk.3.ffn_gate_inp.weight"), None, "26B-A4B");
        assert_eq!(with("blk.3.ffn_down_exps.scale"), None);
        for scale in [
            "blk.3.ffn_gate_exps.scale",
            "blk.3.ffn_up_exps.scale",
            "blk.3.ffn_gate_up_exps.scale",
        ] {
            assert!(refused(with(scale), "gate or up projection"), "{scale}");
        }
        assert!(refused(
            with("per_layer_token_embd.weight"),
            "per-layer embeddings"
        ));
    }
}

#[cfg(test)]
pub(super) mod sliding_window_tests {
    use super::*;

    /// A header naming `arch` and the keys given — shared with `split::rope`'s,
    /// `split::hybrid`'s and `split::kv_budget`'s tests, which read the same keys.
    pub(crate) fn header(arch: &str, extra: Vec<(&str, gguf_file::Value)>) -> gguf_file::Content {
        let mut metadata = HashMap::new();
        metadata.insert(
            "general.architecture".to_string(),
            gguf_file::Value::String(arch.to_string()),
        );
        for (k, v) in extra {
            metadata.insert(k.to_string(), v);
        }
        gguf_file::Content {
            magic: gguf_file::VersionedMagic::GgufV3,
            metadata,
            tensor_infos: HashMap::new(),
            tensor_data_offset: 0,
        }
    }

    fn windowed_layers(w: &SlidingWindow, layers: usize) -> Vec<usize> {
        (0..layers)
            .filter(|&il| w.for_layer(il).is_some())
            .collect()
    }

    /// Gemma 2 as llama.cpp reads it: the file's window on every EVEN layer
    /// (`set_swa_pattern(2)`), 4,096 when the file names none.
    #[test]
    fn gemma2_windows_every_other_layer_from_the_first() {
        let w = SlidingWindow::read(&header(
            "gemma2",
            vec![(
                "gemma2.attention.sliding_window",
                gguf_file::Value::U32(4096),
            )],
        ))
        .expect("gemma2 has a window");
        assert_eq!(w.size, 4096);
        assert_eq!(windowed_layers(&w, 6), vec![0, 2, 4]);
        assert_eq!(w.for_layer(1), None);

        let unnamed = SlidingWindow::read(&header("gemma2", vec![])).unwrap();
        assert_eq!(unnamed.size, 4096, "the family default");
    }

    /// The pattern key wins over the family's period: an integer is a period
    /// (`il % n < n - 1`, 0 = every layer), an array one flag per layer.
    #[test]
    fn a_declared_pattern_overrides_the_family_period() {
        let period6 = SlidingWindow::read(&header(
            "gemma2",
            vec![(
                "gemma2.attention.sliding_window_pattern",
                gguf_file::Value::U32(6),
            )],
        ))
        .unwrap();
        assert_eq!(
            windowed_layers(&period6, 12),
            vec![0, 1, 2, 3, 4, 6, 7, 8, 9, 10]
        );

        let every = SlidingWindow::read(&header(
            "gemma2",
            vec![(
                "gemma2.attention.sliding_window_pattern",
                gguf_file::Value::U32(0),
            )],
        ))
        .unwrap();
        assert_eq!(windowed_layers(&every, 3), vec![0, 1, 2]);

        let flags = SlidingWindow::read(&header(
            "gemma2",
            vec![(
                "gemma2.attention.sliding_window_pattern",
                gguf_file::Value::Array(vec![
                    gguf_file::Value::Bool(false),
                    gguf_file::Value::Bool(true),
                    gguf_file::Value::Bool(true),
                ]),
            )],
        ))
        .unwrap();
        assert_eq!(
            windowed_layers(&flags, 5),
            vec![1, 2],
            "past the list: none"
        );
    }

    /// Gemma 3 as llama.cpp reads it (`llama_model_gemma3`): five layers in
    /// six — all but every sixth — and only when the file names a window, since
    /// the family has no default.
    #[test]
    fn gemma3_windows_five_layers_in_six_only_when_the_file_names_a_window() {
        let w = SlidingWindow::read(&header(
            "gemma3",
            vec![(
                "gemma3.attention.sliding_window",
                gguf_file::Value::U32(512),
            )],
        ))
        .expect("the 1B names 512");
        assert_eq!(w.size, 512);
        assert_eq!(windowed_layers(&w, 12), vec![0, 1, 2, 3, 4, 6, 7, 8, 9, 10]);
        assert_eq!(SlidingWindow::read(&header("gemma3", vec![])), None);
    }

    fn gemma_shape(
        arch: &str,
        layers: u32,
        n_embd: u32,
        n_head: u32,
        key_length: u32,
    ) -> gguf_file::Content {
        let key = |k: &str| format!("{arch}.{k}");
        let (a, b, c, d) = (
            key("block_count"),
            key("embedding_length"),
            key("attention.head_count"),
            key("attention.key_length"),
        );
        header(
            arch,
            vec![
                (a.as_str(), gguf_file::Value::U32(layers)),
                (b.as_str(), gguf_file::Value::U32(n_embd)),
                (c.as_str(), gguf_file::Value::U32(n_head)),
                (d.as_str(), gguf_file::Value::U32(key_length)),
            ],
        )
    }

    fn gemma2_shape(layers: u32, n_embd: u32, n_head: u32, key_length: u32) -> gguf_file::Content {
        gemma_shape("gemma2", layers, n_embd, n_head, key_length)
    }

    /// Gemma 3's 27B (62 layers, 5376 / 32 = 168 against a head of 128) is
    /// scaled by the model width; the 1B and 4B, whose width ratios also
    /// differ from their heads (288 and 320 against 256), by the head.
    #[test]
    fn only_gemma3_27b_scales_its_queries_by_the_model_width() {
        let scale =
            query_scale(&gemma_shape("gemma3", 62, 5376, 32, 128), 128).expect("the 27B is scaled");
        assert!((scale - (128f64 / 168f64).sqrt()).abs() < 1e-12, "{scale}");
        assert_eq!(
            query_scale(&gemma_shape("gemma3", 34, 2560, 8, 256), 256),
            None,
            "4B"
        );
        assert_eq!(
            query_scale(&gemma_shape("gemma3", 26, 1152, 4, 256), 256),
            None,
            "1B"
        );
    }

    /// The real 12B's attention keys (read off unsloth's Q4_K_M, 2026-10-16):
    /// 48 layers, five in six windowed over 1,024 positions, 256-wide heads and
    /// 8 K/V heads on those against 512-wide and 1 on every sixth. The counts
    /// are a per-layer list, written as INT32 by gguf-py.
    pub(crate) fn gemma4_12b() -> gguf_file::Content {
        use gguf_file::Value::{Array, Bool, F32, I32, U32};
        let windowed = |il: u32| il % 6 != 5;
        header(
            "gemma4",
            vec![
                ("gemma4.block_count", U32(48)),
                ("gemma4.embedding_length", U32(3840)),
                ("gemma4.attention.head_count", U32(16)),
                (
                    "gemma4.attention.head_count_kv",
                    Array(
                        (0..48)
                            .map(|il| I32(if windowed(il) { 8 } else { 1 }))
                            .collect(),
                    ),
                ),
                ("gemma4.attention.key_length", U32(512)),
                ("gemma4.attention.key_length_swa", U32(256)),
                ("gemma4.rope.dimension_count", U32(512)),
                ("gemma4.rope.dimension_count_swa", U32(256)),
                ("gemma4.rope.freq_base", F32(1_000_000.0)),
                ("gemma4.rope.freq_base_swa", F32(10_000.0)),
                ("gemma4.attention.sliding_window", U32(1024)),
                (
                    "gemma4.attention.sliding_window_pattern",
                    Array((0..48).map(|il| Bool(windowed(il))).collect()),
                ),
            ],
        )
    }

    /// Each layer's shape is read from its own keys, as llama.cpp reads
    /// `n_head_kv(il)`, `n_embd_head_k(il)` and `n_rot(il)`; a file with one
    /// count and no `_swa` keys gets one shape everywhere.
    #[test]
    fn gemma4_reads_each_layers_attention_shape() {
        let shapes = attention_shapes(&gemma4_12b()).unwrap();
        assert_eq!(shapes.len(), 48);
        let windowed = AttentionShape {
            n_kv_head: 8,
            head_dim: 256,
            rope_dim: 256,
        };
        let global = AttentionShape {
            n_kv_head: 1,
            head_dim: 512,
            rope_dim: 512,
        };
        for (il, shape) in shapes.iter().enumerate() {
            let want = if il % 6 == 5 { global } else { windowed };
            assert_eq!(*shape, want, "layer {il}");
        }

        let one = attention_shapes(&header(
            "llama",
            vec![
                ("llama.block_count", gguf_file::Value::U32(4)),
                ("llama.embedding_length", gguf_file::Value::U32(4096)),
                ("llama.attention.head_count", gguf_file::Value::U32(32)),
                ("llama.attention.head_count_kv", gguf_file::Value::U32(8)),
            ],
        ))
        .unwrap();
        let shape = AttentionShape {
            n_kv_head: 8,
            head_dim: 128,
            rope_dim: 128,
        };
        assert_eq!(one, vec![shape; 4]);
    }

    /// Gemma 4 normalises Q and K per head and scales by nothing
    /// (`f_attention_scale = 1.0`): each layer's queries are multiplied by its
    /// own `sqrt(head_dim)`, which the attention's `1/sqrt(head_dim)` cancels.
    #[test]
    fn gemma4_attention_is_not_scaled_on_any_layer() {
        let ct = gemma4_12b();
        assert_eq!(query_scale(&ct, 256), Some(16.0));
        assert_eq!(query_scale(&ct, 512), Some(512f64.sqrt()));
    }

    /// llama.cpp's sizes: only the 27B (46 layers) scales by `n_embd / n_head`
    /// — 144 against a head of 128. The 9B's `n_embd / n_head` is 224 against
    /// a head of 256, and llama.cpp scales it by the HEAD, so it reads none.
    #[test]
    fn only_gemma2_27b_scales_its_queries_by_the_model_width() {
        let scale = query_scale(&gemma2_shape(46, 4608, 32, 128), 128).expect("the 27B is scaled");
        assert!((scale - (128f64 / 144f64).sqrt()).abs() < 1e-12, "{scale}");
        assert_eq!(
            query_scale(&gemma2_shape(42, 3584, 16, 256), 256),
            None,
            "9B"
        );
        assert_eq!(
            query_scale(&gemma2_shape(26, 2304, 8, 256), 256),
            None,
            "2B"
        );
        assert_eq!(
            query_scale(
                &header(
                    "llama",
                    vec![("llama.block_count", gguf_file::Value::U32(46))]
                ),
                128
            ),
            None
        );
    }

    /// Families whose window llama.cpp does not apply read none: Phi-3
    /// declares one that llama.cpp switches off, and a family with no window
    /// key is unwindowed. A window of 0 is no window.
    #[test]
    fn only_a_window_llama_cpp_applies_is_read() {
        assert_eq!(
            SlidingWindow::read(&header(
                "phi3",
                vec![("phi3.attention.sliding_window", gguf_file::Value::U32(2047))],
            )),
            None
        );
        assert_eq!(SlidingWindow::read(&header("llama", vec![])), None);
        assert_eq!(
            SlidingWindow::read(&header(
                "gemma2",
                vec![("gemma2.attention.sliding_window", gguf_file::Value::U32(0))],
            )),
            None
        );
    }
}
