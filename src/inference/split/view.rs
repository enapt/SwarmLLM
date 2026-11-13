// ── A loaded model serves any run of its layers (FUTURE_WORK #449) ──
//
// A worker that holds `[s..e)` of a model and is asked for `[a..b)` inside it
// builds a VIEW: a `SplitModel` over exactly `[a..b)` whose weights are the
// loaded instance's own — candle's tensors and quantized matrices are
// reference-counted, so a layer's clone is a handful of pointers — with the
// token table only where `a` is the model's first layer, the head only where
// `b` is its last, and everything a forward keeps per model of its own: the
// KV cache's key ("{a}-{b}-{total}"), the decode graphs, the masks. The
// worker files it under its own key, so every path that keys by range — the
// KV store, the prefix cache, admission, the graphs — runs it unchanged.
//
// Prior art: Petals (`petals/server/handler.py`, read 2026-11-13). A server
// keeps one backend per hosted block; a request names the blocks it wants as
// a chain of block ids, `_check_uids` refuses any it does not host, and
// `rpc_inference` runs exactly those (`requested_backends`) with an attention
// cache allocated for those blocks alone (`_allocate_cache(requested_backends,
// …)`) — never a reload. Taken: a request for a run inside a loaded range is
// served from it, and its cache is its own, sized by its own layers. Before
// this, a computer with a model loaded whole that a split planned `[0..25)`
// read those layers from disk again and held them twice (report #021; the
// 14B's real-link split, 2026-11-13: first token 4.4-4.6 s against 0.4-0.6).

use super::kv_budget::KvGrowth;
use super::model::SplitModel;
use super::LayerVariant;

/// Whether a worker serves a run inside a range it loaded from that range's
/// own weights (#449). On by default; `SWARMLLM_ELASTIC_SPANS=0` is the old
/// rule — every range a forward names is loaded as itself — for an A/B. Read
/// once a process: the worker builds views by it, the daemon charges by it
/// (`process_pool::WorkerHandle::serves_segment`) and the health monitor
/// advertises it (`features::SERVES_SLICES`), so a peer is told the truth.
pub(crate) fn elastic_spans_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("SWARMLLM_ELASTIC_SPANS").as_deref() != Ok("0"))
}

/// Does a loaded `held` range serve `range` from its own weights — inside it
/// and not equal to it (an equal range is the instance itself)?
pub(crate) fn serves_inside(held: (usize, usize), range: (usize, usize)) -> bool {
    held.0 <= range.0 && range.1 <= held.1 && range.0 < range.1 && held != range
}

/// Where a view's weights come from: the range of the instance that loaded
/// them, and the view's place in the order views were built — the oldest idle
/// one is the first to go when a copy has [`MOST_VIEWS_A_COPY`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ViewOf {
    pub(crate) root: (usize, usize),
    pub(crate) built: u64,
}

/// How many views one loaded copy keeps at once. A view costs no weights, but
/// what a forward keeps per model is its own — on a card each decode graph
/// holds its working memory while the view lives — and the daemon charges a
/// view nothing (`WorkerHandle::serves_segment`), so a peer naming run after
/// run inside a loaded range must not grow it without end. A split names a few
/// runs of a model; eight leaves room for several coordinators' plans at once.
/// Past it the oldest view no conversation is using goes (rebuilt on its next
/// forward; its caches stay, keyed by its range), and where every one is in
/// use the run is refused.
///
/// Prior art: vLLM records graphs only for a fixed list of batch sizes
/// (`cudagraph_capture_sizes`), so their memory is bounded before the first
/// request; NVIDIA's forum (threads 294148, 244901, read 2026-11-13): a
/// graph's memory nodes and per-node records stay with it, and destroying one
/// does not hand the memory back at once. Taken: a fixed bound per copy, idle
/// views going first.
pub(crate) const MOST_VIEWS_A_COPY: usize = 8;

impl SplitModel {
    /// The range this instance was built from, where it is a view of another
    /// (`None` for an instance loaded from disk). A view goes with that
    /// instance: it holds the same weights, so keeping it would keep their
    /// memory after the daemon has released it.
    pub(crate) fn viewed_from(&self) -> Option<(usize, usize)> {
        self.viewed_from.map(|v| v.root)
    }

    /// Where this view stands in the order views were built (`None` for an
    /// instance loaded from disk).
    pub(crate) fn view_built(&self) -> Option<u64> {
        self.viewed_from.map(|v| v.built)
    }

    /// A model over exactly `[start..end)` that runs this instance's own
    /// weights, or `None` where it cannot:
    ///
    /// - the range is not inside this one, or is this one;
    /// - a layer reads another layer's cache (`LayerWeights::kv_source`) or the
    ///   model builds per-layer inputs from its tokens (Gemma 4 E2B/E4B) —
    ///   those run only whole (`GgufTensorMeta::runs_only_whole`), so no split
    ///   names a part of them, and their loader refuses one.
    ///
    /// A run starting or ending the model takes this instance's token table
    /// or head as it is: where it has none, a load of the run from the same
    /// disk would have none either (`shard::compute_first_last`), so the
    /// daemon, which counts the run as served without a load
    /// (`WorkerHandle::serves_segment`), and the worker agree.
    ///
    /// The caller asks only of an instance whose weights were not cut for
    /// tensor parallelism (`tp_size == 1`).
    ///
    /// What a view does NOT share: the KV cache (its own key), the decode
    /// graphs, the masks, the reply's counters — and no guess head, block
    /// drafter or stand-in prior, each of which belongs to a whole model or to
    /// a first segment loaded beside the head's part.
    pub(crate) fn view_of(&self, start: usize, end: usize) -> Option<SplitModel> {
        if !serves_inside((self.layer_start, self.layer_end), (start, end)) {
            return None;
        }
        if self.per_layer_inputs.is_some()
            || self
                .layers
                .iter()
                .any(|l| matches!(l, LayerVariant::Dense(w) if w.kv_source.is_some()))
        {
            return None;
        }
        let starts_model = start == 0;
        let ends_model = end == self.total_layers;
        let from = start - self.layer_start;
        let to = end - self.layer_start;
        let layer_devices = if self.layer_devices.is_empty() {
            Vec::new()
        } else {
            self.layer_devices.get(from..to)?.to_vec()
        };
        let (kv_growth, host_kv_growth) = self.view_kv_growth(start, end, &layer_devices);
        Some(SplitModel {
            tok_embeddings: if starts_model {
                self.tok_embeddings.clone()
            } else {
                None
            },
            per_layer_inputs: None,
            layers: self.layers.get(from..to)?.to_vec(),
            norm: if ends_model { self.norm.clone() } else { None },
            output: if ends_model {
                self.output.clone()
            } else {
                None
            },
            mtp: None,
            block_drafter: None,
            drafter_head: None,
            drafter_head_bytes: 0,
            drafter_hot: None,
            plain_steps: Default::default(),
            masks: None,
            image_spans: Vec::new(),
            draft_tree: None,
            layer_start: start,
            layer_end: end,
            total_layers: self.total_layers,
            hidden_dim: self.hidden_dim,
            arch: self.arch.clone(),
            device: self.device.clone(),
            layer_devices,
            vocabulary: self.vocabulary.clone(),
            tokenizer: self.tokenizer.clone(),
            eos_tokens: self.eos_tokens.clone(),
            chat_template: self.chat_template.clone(),
            bos_token: self.bos_token.clone(),
            eos_token: self.eos_token.clone(),
            max_seq_len: self.max_seq_len,
            // One budget for the memory both draw on: `claim_room` weighs the
            // whole store's allocation in a home against it, every model's
            // conversations included, so a view adds no room it does not have.
            kv_budget_bytes: self.kv_budget_bytes,
            kv_growth,
            host_kv_budget_bytes: self.host_kv_budget_bytes,
            host_kv_growth,
            kv_layout: self.kv_layout.clone(),
            kv_model_key: format!("{start}-{end}-{}", self.total_layers),
            final_logit_softcap: self.final_logit_softcap,
            suppressed_logits: if ends_model {
                self.suppressed_logits.clone()
            } else {
                None
            },
            batch_calls: 0,
            batch_fellback: 0,
            batch_stats_reported_at: None,
            decode_graph: Default::default(),
            // The widest copying layer's activations and the expert stacks of
            // the whole instance: a run inside it copies no more than that, so
            // its first prompt forward is sized at most a little generously,
            // and every later one from what the card measured.
            prompt_room: super::model::PromptRoom {
                last: None,
                ..self.prompt_room
            },
            stand_in_prior: None,
            grammar_vocab: Default::default(),
            viewed_from: Some(ViewOf {
                root: self
                    .viewed_from()
                    .unwrap_or((self.layer_start, self.layer_end)),
                built: {
                    static BUILT: std::sync::atomic::AtomicU64 =
                        std::sync::atomic::AtomicU64::new(0);
                    BUILT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
                },
            }),
        })
    }

    /// What a conversation's cache costs over `[start..end)` on this
    /// instance's devices, as (the main device's growth, the processor side's)
    /// — the loader's own reading (`kv_budget::KvLayout::growth`, the ONE
    /// price of a cache), each layer on the device it was placed on. Without
    /// a layout (a test fixture) the instance's own figures, which charge a
    /// run inside it no less than it costs.
    fn view_kv_growth(
        &self,
        start: usize,
        end: usize,
        layer_devices: &[candle_core::Device],
    ) -> (KvGrowth, KvGrowth) {
        let Some(layout) = self.kv_layout.as_deref() else {
            return (self.kv_growth, self.host_kv_growth);
        };
        // A growth is charged only where the loader charged one: a model with
        // no budget on a device has no growth there either.
        let main_charged = self.kv_budget_bytes.is_some();
        let host_charged = self.host_kv_budget_bytes.is_some();
        let on_card = |d: &candle_core::Device| d.is_cuda();
        if layer_devices.is_empty() {
            let main = if main_charged {
                layout.growth(start, end, on_card(&self.device))
            } else {
                KvGrowth::default()
            };
            return (main, KvGrowth::default());
        }
        // Per run of layers on one device, in order.
        let (mut main, mut host) = (KvGrowth::default(), KvGrowth::default());
        let mut run_from = 0usize;
        for i in 1..=layer_devices.len() {
            let ends_run = i == layer_devices.len()
                || on_card(&layer_devices[i]) != on_card(&layer_devices[run_from]);
            if !ends_run {
                continue;
            }
            let card = on_card(&layer_devices[run_from]);
            let growth = layout.growth(start + run_from, start + i, card);
            // The main device is where the segment's caches go unless a layer
            // is placed off it: a card/processor split's processor-held
            // layers keep theirs in host memory (#262).
            if card == on_card(&self.device) {
                if main_charged {
                    main = main.plus(growth);
                }
            } else if host_charged {
                host = host.plus(growth);
            }
            run_from = i;
        }
        (main, host)
    }
}

/// The loaded weights a view runs are its parent's — a test's way to say so
/// without reaching into the layers: the first layer's attention output
/// projection is the same allocation in both.
#[cfg(test)]
pub(crate) fn shares_weights_with(view: &SplitModel, parent: &SplitModel) -> bool {
    let Some(from) = view.layer_start.checked_sub(parent.layer_start) else {
        return false;
    };
    #[derive(PartialEq)]
    enum Allocation {
        Quantized(usize),
        Dense(candle_core::TensorId),
    }
    let wo = |l: &LayerVariant| match l {
        LayerVariant::Dense(w) => match &w.attention_wo.inner {
            crate::inference::layers::QMatMulInner::Standard(
                candle_core::quantized::QMatMul::QTensor(q),
            ) => Some(Allocation::Quantized(std::sync::Arc::as_ptr(q) as usize)),
            crate::inference::layers::QMatMulInner::Standard(
                candle_core::quantized::QMatMul::Tensor(t)
                | candle_core::quantized::QMatMul::TensorF16(t),
            ) => Some(Allocation::Dense(t.id())),
            _ => None,
        },
        _ => None,
    };
    match (view.layers.first(), parent.layers.get(from)) {
        (Some(v), Some(p)) => wo(v).is_some_and(|a| wo(p) == Some(a)),
        _ => false,
    }
}
