//! Cross-request prefix KV-cache.
//!
//! Stores per-model snapshots of the full KV-cache state captured at the end
//! of prompt prefill. On a subsequent request, looks up the longest cached
//! token prefix that is a strict prefix of the new prompt's tokens; on hit,
//! materialises a fresh `KvCacheEntry` seeded with the cached tensors so
//! prefill skips those positions and only processes the suffix.
//!
//! Scope:
//! - Per-worker-subprocess, per-model-key. One instance manages all models
//!   inside a worker.
//! - Flat set of entries per model, bounded by `max_entries`. No radix tree.
//! - **One snapshot per insert, at the full prompt length.** Partial-prefix
//!   reuse comes from `lookup` narrowing a longer entry to the shared prefix
//!   at hit time — the same primitive `export_snapshot_bytes` has always used
//!   to serve any block boundary from a full-length entry. Snapshotting every
//!   block boundary at insert time was gotcha #312: `O(prompt² / block)`
//!   copied bytes per request — ~15 GB and minutes of CPU stall for one
//!   2.9k-token prompt on a 3B model.
//! - LRU eviction by `last_hit` timestamp.
//! - Tensors are cloned on restore (candle `append` copies into a fresh
//!   pre-allocated buffer). No reference-counted sharing across live
//!   requests.
//! - Cross-node sharing: `export_snapshot_bytes` serves any hashed block
//!   boundary to a peer; `hydrate_request_from_bytes` seeds a request from a
//!   peer's serialized snapshot.
//!
//! What this does NOT do:
//! - SSM / hybrid-model state (Qwen3.5-SSM). Caches containing SSM state
//!   are skipped; the model runs a full prefill.
//! - True radix-tree de-duplication. Storage is `O(entries * prompt_len *
//!   hidden * layers)` in the worst case — configure `max_entries` with
//!   this in mind.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};
use std::time::Instant;

use crate::inference::layers::{SsmSlot, SsmState};
use crate::inference::split::kv_cache::LayerKv;
use candle_core::{Device, Tensor};

use crate::error::SwarmError;
use crate::types::{KeepAsked, PrefixBlockEntry};

use super::kv_cache::{CacheLayout, KvCacheEntry, KvCacheStore};

/// Compute the chained BLAKE3 hash chain over a token sequence, in blocks of
/// `block_size` tokens. Returns one entry per block boundary that lies inside
/// `tokens` (i.e. positions `block_size, 2*block_size, ...`). The trailing
/// partial block (if any) is NOT hashed — only complete blocks count, so two
/// peers running the same `block_size` will agree on every hash value for
/// every shared prefix.
///
/// Each block's hash is `blake3(prev_hash || u32_le(tokens[i*B..(i+1)*B]))`,
/// with `prev_hash` empty for the first block. This gives prefix-property:
/// two prompts that share the first `k` blocks have identical `block_hash[0..k]`.
pub fn compute_block_hashes(tokens: &[u32], block_size: usize) -> Vec<PrefixBlockEntry> {
    if block_size == 0 {
        return Vec::new();
    }
    let block_count = tokens.len() / block_size;
    let mut out = Vec::with_capacity(block_count);
    let mut prev: Option<[u8; 32]> = None;
    for i in 0..block_count {
        let mut hasher = blake3::Hasher::new();
        if let Some(p) = prev {
            hasher.update(&p);
        }
        let start = i * block_size;
        for &tok in &tokens[start..start + block_size] {
            hasher.update(&tok.to_le_bytes());
        }
        let h: [u8; 32] = hasher.finalize().into();
        prev = Some(h);
        out.push(PrefixBlockEntry {
            block_hash: h,
            token_count: ((i + 1) * block_size) as u32,
        });
    }
    out
}

/// What a snapshot of `layers` costs: `per_position × pos − slid_away` bytes
/// for one taken at `pos`. Read off the live cache's views, so it is exact for
/// this model on this device and copies nothing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SnapshotCost {
    /// Bytes one position of `layers`' KV occupies, summed over layers. 0 when
    /// nothing is populated. A slid layer's views hold only its positions from
    /// `held_from` (FUTURE_WORK #281), so they are divided by that count — by
    /// the conversation's length they read lighter than they are.
    per_position: usize,
    /// The bytes of the positions slid layers no longer hold: each layer's
    /// per-position bytes × its `held_from`. A snapshot copies each layer from
    /// there (`snapshot_at`), and is never taken short of `serves_from`, which
    /// no layer's `held_from` passes — so a windowed layer costs a snapshot
    /// the positions it holds, never the conversation's length.
    slid_away: usize,
}

impl SnapshotCost {
    fn of(layers: &[Option<LayerKv>]) -> Self {
        let (mut per_position, mut slid_away) = (0usize, 0usize);
        for kv in layers.iter().flatten() {
            let positions = kv.current_seq_len() - kv.held_from();
            if positions == 0 {
                continue;
            }
            let bytes = [kv.k(), kv.v()]
                .into_iter()
                .filter_map(|t| t.ok().flatten())
                .map(|t| t.elem_count() * t.dtype().size_in_bytes() / positions)
                .sum::<usize>();
            per_position = per_position.saturating_add(bytes);
            slid_away = slid_away.saturating_add(bytes.saturating_mul(kv.held_from()));
        }
        Self {
            per_position,
            slid_away,
        }
    }

    /// Bytes of a snapshot taken at `pos`.
    fn bytes_at(self, pos: usize) -> usize {
        self.per_position
            .saturating_mul(pos)
            .saturating_sub(self.slid_away)
    }
}

/// What [`PrefixCache::make_room`] let go of before a copy.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
struct RoomMade {
    /// Entries the new snapshot replaces.
    replaced: usize,
    /// Least recently hit entries evicted so the new one fits the caps.
    evicted: usize,
    /// Whether a replaced entry was kept on this computer (#251 (2b)) — the
    /// new entry inherits it.
    replaced_kept_here: bool,
}

/// Magic bytes identifying a serialized `KvSnapshot` on the wire.
/// `SKVX` = "SwarmLLM KV eXchange". Helps early-reject garbled payloads.
pub const KV_SNAPSHOT_MAGIC: &[u8; 4] = b"SKVX";
/// Snapshot wire format version. Bump when the frame layout changes
/// incompatibly; older nodes reject unknown versions on receive.
pub const KV_SNAPSHOT_VERSION: u32 = 1;

/// BLAKE3-verify that the chained hash over `tokens[..expected_token_count]`
/// at `block_size` granularity produces `expected_hash` at its last block.
/// Returns `true` iff the final chained hash matches — Phase 2 fetchers use
/// this to reject peers that return KV data not matching the requested
/// block hash.
pub fn verify_token_hash_chain(
    tokens: &[u32],
    block_size: usize,
    expected_token_count: usize,
    expected_hash: &[u8; 32],
) -> bool {
    if block_size == 0 || expected_token_count == 0 || expected_token_count > tokens.len() {
        return false;
    }
    if !expected_token_count.is_multiple_of(block_size) {
        // `token_count` must land on a block boundary — the chain hash is
        // only defined at complete-block positions.
        return false;
    }
    let manifest = compute_block_hashes(&tokens[..expected_token_count], block_size);
    match manifest.last() {
        Some(last) => {
            last.token_count as usize == expected_token_count && last.block_hash == *expected_hash
        }
        None => false,
    }
}

/// How far short of a prompt's end a model with recurrent layers saves it
/// (`PrefixCache::save_point`, #228). llama.cpp's server checkpoints such a
/// model 4 tokens before the end, and again a batch earlier and at the last
/// user message (`server-context.cpp`, PR #20288, read at `4b1a27f`); this
/// node saves ONE point, so a template whose opening of a reply is longer
/// than its writing of a past one would miss the whole prompt — 12 more
/// tokens a hit is the cheaper side of that.
pub const RECURRENT_SAVE_BEFORE_END: usize = 16;

/// Per-layer KV snapshot at a specific token position.
pub struct KvSnapshot {
    /// Number of tokens covered by this snapshot (== K/V seq dim).
    pub token_count: usize,
    /// One `(K, V)` per layer in the same order as `KvCacheEntry.layers`.
    /// `None` for layers that had no cache at capture time (shouldn't happen
    /// for a completed prefill, but we tolerate it to avoid losing the entire
    /// cache to one missing layer).
    pub layers: Vec<Option<(Tensor, Tensor)>>,
    /// KV cache's per-layer seq dim (candle `KvCache::dim()`). Used when
    /// reconstructing a fresh `KvCache`.
    pub dim: usize,
    /// Pre-allocated capacity of the source KvCache (`max_seq_len`). Reused
    /// when rebuilding so the restored cache has the same headroom as a
    /// freshly-created one.
    pub max_seq_len: usize,
    /// The card each layer goes back to on a restore, for a layer whose copy
    /// was moved OFF it into host RAM (`SnapshotHome::HostRam`, #257); `None`
    /// — or an empty list — restores a layer where its tensors already are.
    pub restore_to: Vec<Option<Device>>,
    /// The position each layer's copy starts at: 0, or where a sliding-window
    /// layer's slid cache started (`LayerKv::held_from`, FUTURE_WORK #281).
    /// Empty — every layer from 0 — for a snapshot of caches that never slid,
    /// and for every one read off the wire, which carries only those
    /// (`export_snapshot_bytes`).
    pub held_from: Vec<usize>,
    /// The shortest prefix this snapshot can be restored at: past its
    /// windowed layers' starts by a window (`LayerKv::serves_from`, vLLM's
    /// sliding-window prefix-hit rule). 0 for any prefix. For a snapshot
    /// carrying recurrent state, its whole length: a summary of `token_count`
    /// tokens cannot be cut back to fewer (#228).
    pub serves_from: usize,
    /// Each recurrent layer's state after `token_count` tokens (Qwen 3.5's
    /// DeltaNet: `(conv_state, recurrent_state)`), by layer like `layers` —
    /// a layer holds a KV pair or a state, never both. A copy, never the live
    /// buffers, which a card's next steps write into (`SsmState`). Kept in
    /// host RAM with `restore_to` saying where it goes back, as for KV. Empty
    /// for a model without recurrent layers and for every snapshot read off
    /// the wire, which never carries one.
    pub ssm: Vec<Option<(Tensor, Tensor)>>,
}

impl KvSnapshot {
    /// Does this snapshot hold a recurrent state? Such a snapshot is restored
    /// whole or not at all, and never leaves this node: the wire format
    /// carries attention layers only.
    pub fn carries_recurrent_state(&self) -> bool {
        self.ssm.iter().any(Option::is_some)
    }

    /// Where layer `i`'s copy starts — see [`Self::held_from`].
    pub fn layer_held_from(&self, i: usize) -> usize {
        self.held_from.get(i).copied().unwrap_or(0)
    }

    /// Does every layer's copy start at position 0 — the only kind the wire
    /// format describes?
    pub fn holds_every_position(&self) -> bool {
        self.held_from.iter().all(|&from| from == 0)
    }
    /// Bytes this snapshot holds, wherever it holds them.
    ///
    /// The number the retention bound actually needs. An entry's size scales
    /// with its prompt, so a count of entries says nothing useful about
    /// memory: at the insert ceiling on a 3B model one entry can reach about
    /// 1.9 GB, and sixteen of them about 30 GB.
    pub fn bytes(&self) -> usize {
        self.layers
            .iter()
            .chain(self.ssm.iter())
            .flatten()
            .map(|(k, v)| {
                k.elem_count() * k.dtype().size_in_bytes()
                    + v.elem_count() * v.dtype().size_in_bytes()
            })
            .sum()
    }

    /// Where this snapshot's copy is kept: host RAM once any layer was moved
    /// off its card, else beside the live cache.
    pub fn home(&self) -> SnapshotHome {
        if self.restore_to.iter().any(Option::is_some) {
            SnapshotHome::HostRam
        } else {
            SnapshotHome::BesideLive
        }
    }

    /// Bytes this snapshot holds in `home` — what that memory's budget is
    /// charged for it.
    pub fn bytes_in(&self, home: SnapshotHome) -> usize {
        if self.home() == home {
            self.bytes()
        } else {
            0
        }
    }
}

/// Where a prefix-cache snapshot's copy is kept (#257).
///
/// **Under WSL2 a card's snapshot is kept in host RAM, as llama.cpp's server
/// keeps its prompt cache (`--cache-ram`, PR #16391).** On the card one held
/// snapshot — 216 MB for a 3,774-token prompt on a 7B — sat in the memory
/// pool's working set, so the NEXT long prompt's first chunk had to grow the
/// pool at its peak (the f16 copy of the first layer's FFN-down weight beside
/// the activations), and growth through `cuMemAllocAsync` under WSL2 took
/// 1.1-6.9 s at 56 h of Windows uptime (nsys, 2026-10-12, v0.3.236): later
/// prompts read at 413-522 tok/s against 1,566 with the cache off. In host
/// RAM the pool keeps the working set a prompt pass needs.
///
/// **The price is paid at ANY uptime, the stall only at a high one**: on a
/// fresh boot (1.4 h, 2026-10-13) neither arm stalled, and the copy off the
/// card took 59-145 ms per 264 MB snapshot (924 ms at worst) inside the
/// request. So it is taken only where the stall is known to happen
/// (`cuda_pool::card_is_reached_through_wsl`); llama.cpp pays the same copy
/// on a request's path, lazily, when a slot takes a new prompt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SnapshotHome {
    /// On the device each layer runs on, beside the live cache, and charged
    /// against that device's KV budget (gotcha #440): every snapshot on the
    /// processor, and on a card outside WSL2 or under
    /// `SWARMLLM_PREFIX_CACHE_HOME=card`.
    BesideLive,
    /// In host RAM, copied off the card. The card's KV budget is charged
    /// nothing for it; the machine's memory bounds it instead
    /// (`kv_budget::host_snapshot_budget`).
    HostRam,
}

impl SnapshotHome {
    /// The one answer to where a snapshot of a model whose primary device is
    /// `device` is kept.
    pub fn for_device(device: &Device) -> Self {
        Self::choose(
            device.is_cuda(),
            crate::inference::cuda_pool::card_is_reached_through_wsl(),
            home_override(),
        )
    }

    /// Only a CUDA card moves its copy — on the processor the live cache
    /// already IS host RAM, and a unified-memory device would copy into the
    /// same memory it came from — and by default only under WSL2.
    fn choose(on_card: bool, through_wsl: bool, forced: Option<Self>) -> Self {
        match forced {
            Some(home) if on_card => home,
            _ if on_card && through_wsl => Self::HostRam,
            _ => Self::BesideLive,
        }
    }

    /// May a snapshot of a model on `device` go to host RAM when the card has
    /// no room for it beside the live cache (`model_worker::snapshot_room`)?
    /// A card's only, and not under `SWARMLLM_PREFIX_CACHE_HOME=card`, the arm
    /// that keeps every copy on the card.
    pub fn may_leave_the_card(device: &Device) -> bool {
        device.is_cuda() && home_override() != Some(Self::BesideLive)
    }
}

/// `SWARMLLM_PREFIX_CACHE_HOME=card` keeps a card's snapshots on the card (the
/// behaviour before #257) and `=host` moves them to host RAM on any platform —
/// both directions of the A/B inside one binary. Read once.
fn home_override() -> Option<SnapshotHome> {
    use std::sync::OnceLock;
    static HOME: OnceLock<Option<SnapshotHome>> = OnceLock::new();
    *HOME.get_or_init(
        || match std::env::var("SWARMLLM_PREFIX_CACHE_HOME").ok()?.trim() {
            "card" => Some(SnapshotHome::BesideLive),
            "host" => Some(SnapshotHome::HostRam),
            _ => None,
        },
    )
}

/// What one snapshot may take: where it is kept and how many positions fit
/// there. Planned by the worker (`model_worker::snapshot_room`) and REQUIRED
/// by both inserts, so the memory a snapshot was sized against and the memory
/// it lands in are one decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SnapshotRoom {
    pub home: SnapshotHome,
    /// Positions that fit in `home`; 0 stores nothing, `usize::MAX` means no
    /// bound is known.
    pub max_positions: usize,
    /// Bytes of other snapshots kept in `home` — any model's, least recently
    /// hit first — to let go before the copy, beyond the entries it replaces
    /// ([`LetGo`]). The insert lets them go in [`PrefixCache::make_room`],
    /// only once a copy is certain: planned with an eviction of its own, a
    /// prompt already saved, or a copy then sent to the other home, gave
    /// other conversations' prompts up for nothing (#367 (c)).
    pub evict_bytes: u64,
}

/// Bytes of the snapshots a save will let go of because the new one covers
/// them — a growing conversation's previous turn — by where each is kept.
/// Room the plan counts as its own: still held while it is planned, so
/// charged to the memory it is in, but gone before the copy lands (#367 (c);
/// counted as held, a tight host budget evicted ANOTHER conversation's prompt
/// to make room this one's previous turn was about to give back).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct LetGo {
    pub beside_live: u64,
    pub host_ram: u64,
}

impl LetGo {
    fn of<'a>(entries: impl Iterator<Item = &'a Entry>) -> Self {
        entries.fold(Self::default(), |sum, e| Self {
            beside_live: sum.beside_live + e.snapshot.bytes_in(SnapshotHome::BesideLive) as u64,
            host_ram: sum.host_ram + e.snapshot.bytes_in(SnapshotHome::HostRam) as u64,
        })
    }

    /// The bytes let go of in `home`.
    pub fn in_home(self, home: SnapshotHome) -> u64 {
        match home {
            SnapshotHome::BesideLive => self.beside_live,
            SnapshotHome::HostRam => self.host_ram,
        }
    }
}

/// How an insert asks for its [`SnapshotRoom`]: once it knows what the save
/// replaces, and only when a copy is still wanted — the plan is handed the
/// [`LetGo`]. Production callers pass the worker's planner as a closure; a
/// fixed room exists only for tests, so no caller can size a snapshot
/// without being told what it replaces.
pub trait RoomPlan {
    fn plan(self, let_go: LetGo) -> SnapshotRoom;
}

impl<F: FnOnce(LetGo) -> SnapshotRoom> RoomPlan for F {
    fn plan(self, let_go: LetGo) -> SnapshotRoom {
        self(let_go)
    }
}

#[cfg(test)]
impl RoomPlan for SnapshotRoom {
    fn plan(self, _let_go: LetGo) -> SnapshotRoom {
        self
    }
}

struct Entry {
    /// The prompt's tokens — empty for an entry keyed by a block chain.
    tokens: Vec<u32>,
    /// A split segment's entry (FUTURE_WORK #10): the coordinator's block
    /// keys for the prompt it holds, one per `chain_block` tokens — a segment
    /// past the first is sent hidden states, never the tokens. Empty for a
    /// token-keyed entry. Every token-path walk passes these by: an empty
    /// `tokens` matches no prompt and covers none.
    chain: Vec<[u8; 32]>,
    chain_block: u32,
    snapshot: Arc<KvSnapshot>,
    /// Logical clock tick of the most recent hit/insert. Used for LRU
    /// eviction (`bucket.sort_by_key`). Atomic so cache hits stay on the
    /// read-lock path — bumping the timestamp no longer needs a write
    /// lock upgrade.
    last_hit: AtomicU64,
    /// Saved from a request whose caller asked to keep it on this computer
    /// (#251 (2b)): reused here, never announced to a peer or exported to one
    /// — `manifest` is empty for it, and `export_snapshot_bytes` finds an
    /// entry only through `manifest`.
    kept_here: bool,
}

impl Entry {
    /// Does this entry already hold `tokens` (`at` = their length) — so a
    /// save of them copies nothing (`lookup` narrows a longer entry)?
    fn covers_tokens(&self, tokens: &[u32], at: usize) -> bool {
        self.tokens.starts_with(tokens) && self.snapshot.serves_from <= at
    }

    /// Does a save of `tokens`, answering from `serves`, replace this entry?
    /// A chain-keyed entry has empty `tokens`, which every prompt "starts
    /// with" — it is never replaced by a token entry. Nor is one shorter than
    /// where a slid snapshot answers from: it serves an earlier branch of the
    /// conversation the new one cannot — llama.cpp keeps checkpoints of a
    /// windowed cache for the same reason (FUTURE_WORK #281).
    fn replaced_by_tokens(&self, tokens: &[u32], serves: usize) -> bool {
        self.chain.is_empty() && tokens.starts_with(&self.tokens) && self.tokens.len() >= serves
    }

    /// Does this split entry already hold `chain` (`positions` long)?
    fn covers_chain(&self, chain: &[[u8; 32]], block_tokens: u32, positions: usize) -> bool {
        self.chain_block == block_tokens
            && self.chain.starts_with(chain)
            && self.snapshot.token_count >= positions
            && self.snapshot.serves_from <= positions
    }

    /// Does a save of `chain`, answering from `serves`, replace this split
    /// entry — the one of this split that ours extends, unless it is shorter
    /// than where ours answers from?
    fn replaced_by_chain(&self, chain: &[[u8; 32]], block_tokens: u32, serves: usize) -> bool {
        !self.chain.is_empty()
            && self.chain_block == block_tokens
            && chain.starts_with(&self.chain)
            && self.snapshot.token_count >= serves
    }

    /// Compute the chained BLAKE3 manifest for this entry's tokens at the
    /// configured block size. Only complete blocks contribute; the entry's
    /// trailing partial block (if any, when `tokens.len() % block_size != 0`)
    /// is omitted so cross-peer block boundaries align deterministically.
    ///
    /// Empty for a snapshot of a slid cache (FUTURE_WORK #281): the wire
    /// format carries every position from 0, and a peer reading a partial
    /// layer as whole would answer from the wrong keys — so it is neither
    /// announced nor exported. Empty too for an entry kept on this computer
    /// (#251 (2b)): a block hash is a commitment to the prompt it was taken
    /// from, and the snapshot is the prompt's cache. Empty for a snapshot
    /// carrying recurrent state (#228): the wire format has no place for it,
    /// and a peer restoring the attention layers alone would answer from a
    /// state that starts over.
    fn manifest(&self, block_size: usize) -> Vec<PrefixBlockEntry> {
        if self.kept_here
            || !self.snapshot.holds_every_position()
            || self.snapshot.carries_recurrent_state()
        {
            return Vec::new();
        }
        compute_block_hashes(&self.tokens, block_size)
    }
}

struct Inner {
    /// Per-model entries keyed by `SplitModel::kv_model_key()`.
    per_model: HashMap<String, Vec<Entry>>,
}

/// Flat, longest-prefix prefix KV-cache shared across requests on a worker.
pub struct PrefixCache {
    inner: RwLock<Inner>,
    /// Monotonic logical clock for LRU ordering. Bumped on every hit and
    /// insert; the value is stored on the touched entry's `last_hit`
    /// atomic. Lookups can therefore record a hit without upgrading from
    /// read to write lock.
    clock: AtomicU64,
    /// Maximum entries retained per model. Older entries (by `last_hit`)
    /// are evicted when the cap is exceeded.
    ///
    /// A SECONDARY bound. It caps the linear lookup walk, not memory — see
    /// `max_bytes`, which is the one that expresses what actually needs
    /// bounding.
    max_entries: usize,
    /// Maximum bytes of KV retained per model, or 0 for no byte bound.
    ///
    /// The bound that matters. Entry size scales with prompt length, so a
    /// count cap does not express memory at all: at the insert ceiling on a 3B
    /// model a single entry can reach about 1.9 GB, and sixteen distinct long
    /// conversations against one model could in principle retain about 30 GB.
    /// Covered-prefix pruning keeps a growing conversation to one entry, so
    /// reaching that needs many DISTINCT long prompts — rare, but nothing
    /// prevented it.
    max_bytes: usize,
    /// Minimum prefix length (tokens) below which lookups return miss and
    /// inserts are skipped. Avoids caching trivial prompts.
    min_tokens: usize,
    /// The most positions ONE snapshot keeps, or 0 for no token ceiling (the
    /// byte budget then decides — see [`Self::positions_ceiling`]). A longer
    /// prompt keeps its OPENING rather than nothing: that is the system prompt
    /// and tool definitions every later turn repeats, which is the part worth
    /// having.
    max_prompt_tokens: usize,
    /// Block granularity for the chained BLAKE3 manifest used by cross-node
    /// prefix sharing. Inserts always store ONE snapshot at the full prompt
    /// length (see gotcha #312 — per-boundary snapshots were quadratic);
    /// blocks only shape the announce manifest and the boundaries
    /// `export_snapshot_bytes` can serve. 0 disables the manifest (no
    /// cross-node announcing); local narrowing lookups are unaffected.
    block_tokens: usize,
    enabled: bool,
}

impl PrefixCache {
    pub fn new(
        enabled: bool,
        max_entries: usize,
        block_tokens: usize,
        min_tokens: usize,
        max_prompt_tokens: usize,
        max_bytes: usize,
    ) -> Self {
        Self {
            inner: RwLock::new(Inner {
                per_model: HashMap::new(),
            }),
            clock: AtomicU64::new(0),
            max_entries,
            max_bytes,
            min_tokens,
            max_prompt_tokens,
            block_tokens,
            enabled,
        }
    }

    /// Bump the logical clock and return the new tick. Used as the
    /// timestamp written to an entry's `last_hit`.
    fn next_tick(&self) -> u64 {
        // Relaxed is fine: we only need monotonicity; total ordering across
        // entries doesn't matter as long as each entry's last_hit reflects a
        // tick from after the operation that touched it.
        self.clock.fetch_add(1, Ordering::Relaxed) + 1
    }

    pub fn enabled(&self) -> bool {
        self.enabled
    }

    /// The most positions one snapshot may keep, given what it costs
    /// (`SnapshotCost`): the explicit token ceiling when one is set, and
    /// never more than the per-model byte budget holds.
    ///
    /// **A slid windowed layer is charged the positions it holds** (#281):
    /// charged every position, Gemma 4 12B's 786 KB a position on a processor
    /// held the default 2 GiB budget to 2,730 positions, and a snapshot never
    /// stops short of where its windowed layers still answer (`serves_from`,
    /// within a growth quantum of the prompt's end) — so no Gemma 4 prompt
    /// past about 3,000 positions was kept at all.
    ///
    /// **Derived rather than a constant, because what a position costs varies
    /// ~3x between models of one size** — 73.7 KB for qwen2.5-3b's two KV
    /// heads, 229 KB for llama-3.2-3b's eight. The fixed 8,192-token ceiling
    /// this replaced refused every longer prompt outright, and agent harnesses
    /// send 4,100-14,400 tokens a turn (field report 2026-09-26), so by
    /// default most agent prompts were never cached at all. Memory is not what
    /// the constant protected: the snapshot is also held to the room beside
    /// the live cache (`max_positions`, gotcha #440), and admission evicts
    /// cached prompts before refusing a request.
    fn positions_ceiling(&self, cost: SnapshotCost) -> usize {
        let by_tokens = if self.max_prompt_tokens == 0 {
            usize::MAX
        } else {
            self.max_prompt_tokens
        };
        let by_bytes = if self.max_bytes == 0 || cost.per_position == 0 {
            usize::MAX
        } else {
            self.max_bytes.saturating_add(cost.slid_away) / cost.per_position
        };
        by_tokens.min(by_bytes)
    }

    /// Block-size used to chunk prompt tokens into chained BLAKE3 hashes for
    /// cross-node prefix-cache sharing. Returns 0 when the manifest is
    /// disabled — callers should treat 0 as "no manifest" and skip
    /// announcing.
    pub fn block_tokens(&self) -> usize {
        self.block_tokens
    }

    /// Find the longest cached token prefix shared with `input_tokens` for
    /// `model_key`. Returns `None` if no suitable prefix is cached or the
    /// cache is disabled.
    ///
    /// An entry no longer has to be a whole prefix of the input: a longer
    /// entry is narrowed to the shared length at hit time — the same
    /// primitive `export_snapshot_bytes` uses to serve any block boundary
    /// from a full-length entry. This is what replaced per-block-boundary
    /// snapshot storage (gotcha #312), and it also means an *identical*
    /// repeated prompt now hits at `len - 1` (one token left to forward)
    /// where it previously missed outright.
    pub fn lookup(&self, model_key: &str, input_tokens: &[u32]) -> Option<Arc<KvSnapshot>> {
        if !self.enabled || input_tokens.len() < self.min_tokens {
            return None;
        }
        // Must keep at least one token to forward — otherwise the model has
        // nothing to compute logits for. Clamp the usable prefix length.
        let usable_max = input_tokens.len().saturating_sub(1);

        // Fast path: read lock only. Walk to find the longest shared prefix,
        // clone the winning snapshot Arc, and bump the entry's atomic
        // last_hit without ever upgrading to a write lock.
        let inner = self.inner.read().ok()?;
        let entries = inner.per_model.get(model_key)?;
        let mut best: Option<(&Entry, usize)> = None;
        for e in entries.iter() {
            let lcp = e
                .tokens
                .iter()
                .zip(input_tokens.iter())
                .take_while(|(a, b)| a == b)
                .count();
            let usable = lcp.min(usable_max);
            // A snapshot of a slid cache cannot be cut back past where its
            // windowed layers still answer (FUTURE_WORK #281).
            if usable < self.min_tokens || usable < e.snapshot.serves_from {
                continue;
            }
            match best {
                None => best = Some((e, usable)),
                Some((_, cur)) if usable > cur => best = Some((e, usable)),
                _ => {}
            }
        }
        let (winner, usable) = best?;
        winner.last_hit.store(self.next_tick(), Ordering::Relaxed);
        let full = winner.snapshot.clone();
        drop(inner);
        // Narrow OUTSIDE the lock — it copies up to `usable` tokens of KV and
        // must not stall concurrent lookups/inserts. Whole-entry hits skip
        // the copy entirely (the Arc is shared; hydrate copies on append).
        let snapshot = if usable == full.token_count {
            full
        } else {
            match narrow_snapshot(&full, usable) {
                Ok(s) => Arc::new(s),
                Err(e) => {
                    tracing::warn!(
                        model_key,
                        usable,
                        entry_tokens = full.token_count,
                        error = %e,
                        "prefix-cache: narrowing cached snapshot failed — treating as miss"
                    );
                    return None;
                }
            }
        };
        tracing::info!(
            model_key,
            matched_tokens = snapshot.token_count,
            prompt_tokens = input_tokens.len(),
            "DIAG: prefix-cache HIT"
        );
        Some(snapshot)
    }

    /// Where a prompt pass saves its opening — the ONE answer for both
    /// generate paths (#228).
    ///
    /// The prompt's end for a model whose caches can be cut back: `lookup`
    /// narrows that snapshot to whatever a later prompt shares. A model with
    /// recurrent layers (Qwen 3.5) is restored whole or not at all, and the
    /// next turn's prompt does not repeat this one's end — the template's
    /// opening of the reply (`<|im_start|>assistant\n<think>\n\n</think>\n\n`)
    /// is not how it writes a past reply — so its pass stops
    /// [`RECURRENT_SAVE_BEFORE_END`] tokens short, is saved there, and goes on.
    pub fn save_point(prompt_tokens: usize, recurrent: bool) -> usize {
        if recurrent {
            prompt_tokens.saturating_sub(RECURRENT_SAVE_BEFORE_END)
        } else {
            prompt_tokens
        }
    }

    /// Capture the current KV state of `kv_store` for `request_id` as ONE
    /// snapshot at the full prompt length and insert it into the cache.
    /// Called after prefill (or at end of decode) when the full prompt KV is
    /// populated.
    ///
    /// One snapshot, not one per block boundary: per-boundary snapshots each
    /// copied `[0..pos]` of every layer's K and V, which is
    /// `O(prompt² / block)` bytes — ~15 GB copied and minutes of CPU stall
    /// for a single 2.9k-token prompt on a 3B model (gotcha #312). `lookup`
    /// and `export_snapshot_bytes` both narrow the full-length snapshot to
    /// whatever boundary a consumer needs, so the extra entries bought
    /// nothing.
    ///
    /// `prompt_tokens` is the token sequence that produced the KV state —
    /// its length must equal the current KV cache seq_len for this request.
    pub fn insert_from_kv(
        &self,
        model_key: &str,
        request_id: &str,
        kv_store: &KvCacheStore,
        prompt_tokens: &[u32],
        // Where the copy goes and what fits there (`model_worker::snapshot_room`
        // → `kv_budget::plan_snapshot` — a snapshot that does not fit is worse
        // than none, gotcha #440), asked once the save knows what it replaces
        // and only if a copy is still wanted (#367 (c)).
        room: impl RoomPlan,
        // The request's own instruction (#251 (2b)) — required, so neither
        // insert site can save a kept request's cache as one to share.
        keep: KeepAsked,
    ) -> Vec<PrefixBlockEntry> {
        // Each bail-out below says why. They were silent, and that made a
        // cross-node prefix-KV measurement undiagnosable: the producing node
        // simply never announced any blocks, with nothing in the log to say
        // which condition stopped it.
        if !self.enabled || prompt_tokens.len() < self.min_tokens {
            tracing::debug!(
                model_key,
                enabled = self.enabled,
                prompt_tokens = prompt_tokens.len(),
                min_tokens = self.min_tokens,
                "prefix-cache: not snapshotting — disabled or prompt below the floor"
            );
            return Vec::new();
        }
        let key = KvCacheStore::cache_key(model_key, request_id);
        let held = match HeldCache::read(kv_store, &key) {
            Ok(held) => held,
            Err(why) => {
                tracing::debug!(
                    model_key,
                    request_id,
                    "prefix-cache: not snapshotting — {why}"
                );
                return Vec::new();
            }
        };
        // Bound insertion points by what the KV actually holds and by the
        // ceiling — which keeps the opening of a longer prompt instead of
        // refusing it — and, below, by the room.
        let ceiling = self.positions_ceiling(held.cost);
        if ceiling < prompt_tokens.len() {
            tracing::info!(
                model_key,
                prompt_tokens = prompt_tokens.len(),
                keeping = ceiling,
                max_prompt_tokens = self.max_prompt_tokens,
                max_mb = self.max_bytes / (1024 * 1024),
                "prefix-cache: prompt longer than one snapshot may hold — keeping its opening"
            );
        }
        let whole = held.positions.min(prompt_tokens.len()).min(ceiling);
        if whole < self.min_tokens {
            tracing::debug!(
                model_key,
                available = whole,
                min_tokens = self.min_tokens,
                "prefix-cache: not snapshotting — fewer KV positions than the floor"
            );
            return Vec::new();
        }
        // Fast path: an entry already covering this prefix (equal or longer)
        // makes the snapshot redundant — `lookup` narrows the longer entry to
        // any shared length at hit time. Skipping here is what makes a
        // repeated identical prompt cost nothing instead of re-copying the
        // whole KV every request — and, asked before the room is planned, give
        // nothing up either.
        if let Some(manifest) = self.covered_by_tokens(model_key, &prompt_tokens[..whole], whole) {
            return manifest;
        }
        let planned_let_go = self.let_go(model_key, |e| {
            e.replaced_by_tokens(&prompt_tokens[..whole], held.serves_once_saved(whole))
        });
        let room = room.plan(planned_let_go);
        if room.max_positions == 0 {
            tracing::info!(
                model_key,
                request_id,
                prompt_tokens = prompt_tokens.len(),
                home = ?room.home,
                "DIAG: prefix-cache: not snapshotting — no room for the copy"
            );
            return Vec::new();
        }
        let available = whole.min(room.max_positions);
        if available < self.min_tokens {
            tracing::debug!(
                model_key,
                available,
                min_tokens = self.min_tokens,
                "prefix-cache: not snapshotting — fewer KV positions than the floor"
            );
            return Vec::new();
        }
        // A sliding-window layer that has dropped the prompt's opening cannot
        // be saved at a length its window no longer reaches — the opening the
        // ceiling or the room would keep (FUTURE_WORK #281).
        if available < held.serves {
            tracing::info!(
                model_key,
                available,
                serves_from = held.serves,
                "prefix-cache: not snapshotting — the sliding-window layers no longer hold \
                 the opening that fits"
            );
            return Vec::new();
        }
        // A recurrent layer's state summarises the positions it has read and
        // cannot be cut back, so it is saved only where the pass stopped to
        // save it (`PrefixCache::save_point`, #228) — never beside attention
        // layers narrowed to a shorter opening.
        if let Some(&at) = held.recurrent_at.iter().find(|&&at| at != available) {
            tracing::info!(
                model_key,
                available,
                recurrent_state_at = at,
                "prefix-cache: not snapshotting — a recurrent layer's state is not at the \
                 position the attention layers would be saved at"
            );
            return Vec::new();
        }

        let snap_tokens = &prompt_tokens[..available];
        // Where the snapshot will answer from (`snapshot_at`'s figure): a
        // recurrent state's serves only its whole length.
        let serves_once_saved = held.serves_once_saved(available);
        if available < whole {
            // Cut to the room: an entry holding more of this prompt covers
            // the cut, and one it no longer replaces was planned as let go.
            if let Some(manifest) = self.covered_by_tokens(model_key, snap_tokens, available) {
                return manifest;
            }
            let let_go = self.let_go(model_key, |e| {
                e.replaced_by_tokens(snap_tokens, serves_once_saved)
            });
            if let_go.in_home(room.home) < planned_let_go.in_home(room.home) {
                tracing::info!(
                    model_key,
                    available,
                    whole,
                    "prefix-cache: not snapshotting — the room's cut keeps an entry the plan \
                     counted as let go"
                );
                return Vec::new();
            }
        }
        let Some(entry) = kv_store_get(kv_store, &key)
            .filter(|e| HeldCache::positions_of(&e.layers) >= available)
        else {
            tracing::debug!(
                model_key,
                request_id,
                "prefix-cache: not snapshotting — the KV entry changed while its room was planned"
            );
            return Vec::new();
        };
        let room_made = self.make_room(
            model_key,
            |e| e.replaced_by_tokens(snap_tokens, serves_once_saved),
            held.cost.bytes_at(available),
            room,
        );
        let copy_started = Instant::now();
        let snap = match snapshot_at(
            &entry.layers,
            &entry.ssm_states,
            available,
            held.dim,
            held.max_seq_len,
            room.home,
        ) {
            Ok(s) => Arc::new(s),
            Err(e) => {
                // Most likely an allocation failure on a memory-starved
                // device (observed on a 6 GB card, 2026-08-02). The cache is
                // an optimisation — degrade to "not cached", never fail the
                // request.
                tracing::warn!(
                    model_key,
                    tokens = available,
                    replaced = room_made.replaced,
                    evicted = room_made.evicted,
                    error = %e,
                    "prefix-cache: snapshot failed — request continues uncached"
                );
                return Vec::new();
            }
        };
        debug_assert_eq!(snap.serves_from, serves_once_saved);
        drop(entry);
        let copy_ms = copy_started.elapsed().as_millis() as u64;
        let (snap_bytes, snap_home) = (snap.bytes(), snap.home());

        let Ok(mut inner) = self.inner.write() else {
            return Vec::new();
        };
        let bucket = inner.per_model.entry(model_key.to_string()).or_default();
        let tokens = snap_tokens.to_vec();
        let tick = self.next_tick();
        // Entries whose tokens are a prefix of ours (including an exact match
        // that raced in since the read-lock check) are fully covered by the
        // new entry — `lookup` narrows to serve them — so retaining them
        // would only burn `max_entries` slots and their KV bytes. A growing
        // conversation therefore keeps ONE entry, not one per turn.
        let serves = snap.serves_from;
        // An entry this one replaces passes its keep on: the strictest
        // instruction wins, as it does for the request (`api::keep_here`).
        let kept_here = keep == KeepAsked::OnThisComputer
            || room_made.replaced_kept_here
            || bucket
                .iter()
                .any(|e| e.replaced_by_tokens(&tokens, serves) && e.kept_here);
        bucket.retain(|e| !e.replaced_by_tokens(&tokens, serves));
        bucket.push(Entry {
            tokens,
            chain: Vec::new(),
            chain_block: 0,
            snapshot: snap,
            last_hit: AtomicU64::new(tick),
            kept_here,
        });
        self.evict_over_caps(bucket);

        let entry_count = bucket.len();
        // Compute the full post-insert manifest for this model so the caller
        // can announce our current cache state (not just the delta). Cheap:
        // BLAKE3 on a few KB of token IDs.
        let manifest = enumerate_manifest_locked(bucket, self.block_tokens);

        tracing::info!(
            model_key,
            entries = entry_count,
            manifest_len = manifest.len(),
            positions = available,
            mb = snap_bytes / (1024 * 1024),
            home = ?snap_home,
            copy_ms = copy_ms,
            replaced_before_copy = room_made.replaced,
            evicted_before_copy = room_made.evicted,
            "DIAG: prefix-cache inserted snapshot"
        );
        manifest
    }

    /// The manifest to announce if an entry of `model_key` already holds
    /// `tokens` (`at` long) — a save of them copies nothing — after marking
    /// that entry hit. `last_hit` is atomic, so the bump needs only the read
    /// lock (same as `lookup`).
    fn covered_by_tokens(
        &self,
        model_key: &str,
        tokens: &[u32],
        at: usize,
    ) -> Option<Vec<PrefixBlockEntry>> {
        let inner = self.inner.read().ok()?;
        let bucket = inner.per_model.get(model_key)?;
        let covering = bucket.iter().find(|e| e.covers_tokens(tokens, at))?;
        covering.last_hit.store(self.next_tick(), Ordering::Relaxed);
        tracing::debug!(
            model_key,
            prompt_tokens = at,
            covering_tokens = covering.tokens.len(),
            "prefix-cache: prefix already covered — skipping snapshot"
        );
        Some(enumerate_manifest_locked(bucket, self.block_tokens))
    }

    /// Blocks held if a split entry of `model_key` already holds `chain`
    /// (`positions` long), after marking it hit.
    fn covered_by_chain(
        &self,
        model_key: &str,
        chain: &[[u8; 32]],
        block_tokens: u32,
        positions: usize,
    ) -> Option<u32> {
        let inner = self.inner.read().ok()?;
        let covering = inner
            .per_model
            .get(model_key)?
            .iter()
            .find(|e| e.covers_chain(chain, block_tokens, positions))?;
        covering.last_hit.store(self.next_tick(), Ordering::Relaxed);
        Some(chain.len() as u32)
    }

    /// What a save lets go of in `model_key`'s bucket — the entries
    /// `replaced` names — by where each is kept ([`LetGo`]).
    fn let_go(&self, model_key: &str, replaced: impl Fn(&Entry) -> bool) -> LetGo {
        self.inner
            .read()
            .ok()
            .and_then(|i| {
                i.per_model
                    .get(model_key)
                    .map(|b| LetGo::of(b.iter().filter(|e| replaced(e))))
            })
            .unwrap_or_default()
    }

    /// Makes room in `model_key`'s bucket for a snapshot of `incoming_bytes`
    /// BEFORE it is copied (#367): first the entries it replaces
    /// (`superseded`), then the least recently hit until it fits both caps —
    /// llama.cpp's `server_prompt_cache::alloc` order ("make room before
    /// allocating the new vectors", `tools/server/server-task.cpp` at
    /// 4b1a27f) — then, if the memory's own budget needs more
    /// (`room.evict_bytes`, from the plan), the least recently hit kept in
    /// `room.home`, any model's. What it lets go is what
    /// [`Self::evict_over_caps`] would drop after the insert; done first, the
    /// save never holds what it replaces beside the copy (a 14B's
    /// 1,800-token prompt: 569 MB), and the copy lands mostly in memory the
    /// allocator already holds instead of fresh pages — under WSL2 a fresh
    /// 569 MB took 255 ms to touch with memory free and 0.6-0.8 s under a
    /// full page cache (the kernel's huge-page compactions failing), against
    /// 43 ms into touched memory; in a turn-by-turn model of the worker
    /// (mimalloc v3, as the workers run it) a save took 171-274 ms beside the
    /// old snapshot and 81-125 ms after letting it go. Each insert's own pass
    /// after the copy stays: it catches an entry that raced in meanwhile. The
    /// cost: a copy that then fails keeps neither — it fails on a device with
    /// no memory left, where that room is what it lacked.
    fn make_room(
        &self,
        model_key: &str,
        superseded: impl Fn(&Entry) -> bool,
        incoming_bytes: usize,
        room: SnapshotRoom,
    ) -> RoomMade {
        // `SWARMLLM_SNAPSHOT_ROOM=after-copy`: the old order — each insert's
        // own pass after the copy drops the replaced entries and the caps'.
        // The plan's own eviction is made before the copy either way, as it
        // always was. Read once.
        static AFTER_COPY: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let after_copy = *AFTER_COPY
            .get_or_init(|| std::env::var("SWARMLLM_SNAPSHOT_ROOM").as_deref() == Ok("after-copy"));
        // Dropped once the lock is released: freeing 0.5 GB is not instant.
        let (gone, room_made) = {
            let Ok(mut inner) = self.inner.write() else {
                return RoomMade::default();
            };
            let mut gone: Vec<Entry> = Vec::new();
            let mut replaced = 0;
            let mut replaced_kept_here = false;
            if let Some(bucket) = inner.per_model.get_mut(model_key).filter(|_| !after_copy) {
                gone = bucket.extract_if(.., |e| superseded(e)).collect();
                replaced = gone.len();
                replaced_kept_here = gone.iter().any(|e| e.kept_here);
                bucket.sort_by_key(|e| e.last_hit.load(Ordering::Relaxed));
                let mut total: usize = bucket.iter().map(|e| e.snapshot.bytes()).sum();
                while !bucket.is_empty()
                    && (bucket.len() >= self.max_entries
                        || (self.max_bytes > 0
                            && total.saturating_add(incoming_bytes) > self.max_bytes))
                {
                    let oldest = bucket.remove(0);
                    total = total.saturating_sub(oldest.snapshot.bytes());
                    gone.push(oldest);
                }
                if bucket.is_empty() {
                    inner.per_model.remove(model_key);
                }
            }
            // The plan's eviction counts what the caps already gave back in
            // the same memory; the replaced entries were its own room.
            let freed_in_home: u64 = gone[replaced..]
                .iter()
                .map(|e| e.snapshot.bytes_in(room.home) as u64)
                .sum();
            if room.evict_bytes > freed_in_home {
                gone.extend(take_least_recently_hit(
                    &mut inner,
                    room.evict_bytes - freed_in_home,
                    room.home,
                ));
            }
            let room_made = RoomMade {
                replaced,
                evicted: gone.len() - replaced,
                replaced_kept_here,
            };
            (gone, room_made)
        };
        drop(gone);
        room_made
    }

    /// Evict LRU entries of one model until both caps hold.
    ///
    /// Count first, because it is cheap and bounds the linear lookup walk.
    /// Then bytes, which is the bound that expresses memory: entries are
    /// sized by their prompt, so staying under sixteen of them says nothing
    /// about how much is being held. Token-keyed and chain-keyed entries share
    /// both caps — one budget for one memory (gotcha #440).
    fn evict_over_caps(&self, bucket: &mut Vec<Entry>) {
        if bucket.len() > self.max_entries {
            bucket.sort_by_key(|e| e.last_hit.load(Ordering::Relaxed));
            let drop_count = bucket.len() - self.max_entries;
            bucket.drain(..drop_count);
        }
        if self.max_bytes > 0 {
            let mut total: usize = bucket.iter().map(|e| e.snapshot.bytes()).sum();
            if total > self.max_bytes {
                bucket.sort_by_key(|e| e.last_hit.load(Ordering::Relaxed));
                // Never evict everything: the entry just inserted is the most
                // recently used, and dropping it would mean this prefill paid
                // to snapshot itself and kept nothing. `positions_ceiling`
                // sizes one snapshot to fit the budget, so a single entry over
                // it arises only where a position's weight could not be read —
                // and then holding it beats holding nothing.
                while total > self.max_bytes && bucket.len() > 1 {
                    let dropped = bucket.remove(0);
                    total = total.saturating_sub(dropped.snapshot.bytes());
                }
            }
        }
    }

    /// Positions `0..want` of a split's prompt, as this segment stored them on
    /// an earlier turn under the coordinator's block chain (FUTURE_WORK #10,
    /// `docs/plans/split_prompt_cache.md`). `want` is the coordinator's
    /// `resume_at` — a whole number of blocks — and nothing shorter will do:
    /// the forward carries only the positions after it. `None` is a miss, which
    /// the caller refuses so the coordinator re-sends the prompt from 0.
    pub fn lookup_chain(
        &self,
        model_key: &str,
        block_tokens: u32,
        keys: &[[u8; 32]],
        want: usize,
    ) -> Option<Arc<KvSnapshot>> {
        let bt = block_tokens as usize;
        if !self.enabled || bt == 0 || want == 0 || !want.is_multiple_of(bt) {
            return None;
        }
        let blocks = want / bt;
        let wanted = keys.get(..blocks)?;
        let inner = self.inner.read().ok()?;
        let winner = inner.per_model.get(model_key)?.iter().find(|e| {
            e.chain_block == block_tokens
                && e.chain.starts_with(wanted)
                && e.snapshot.token_count >= want
                && e.snapshot.serves_from <= want
        })?;
        winner.last_hit.store(self.next_tick(), Ordering::Relaxed);
        let full = winner.snapshot.clone();
        drop(inner);
        // Narrowed outside the lock, as `lookup` does.
        let snapshot = if full.token_count == want {
            full
        } else {
            match narrow_snapshot(&full, want) {
                Ok(s) => Arc::new(s),
                Err(e) => {
                    tracing::warn!(model_key, want, error = %e,
                        "prefix-cache: narrowing a stored split prompt failed — treating as miss");
                    return None;
                }
            }
        };
        tracing::info!(
            model_key,
            restored_positions = want,
            "DIAG: prefix-cache HIT on a split's stored prompt"
        );
        Some(snapshot)
    }

    /// Store this request's cache under a split prompt's block chain, for the
    /// next turn's `lookup_chain`: as many WHOLE blocks as the cache holds,
    /// `room` has space for (from `kv_budget::plan_snapshot` — gotcha #440)
    /// and the byte ceiling allows.
    /// Answers how many blocks it stored — what the coordinator may resume
    /// from here next turn; 0 stores nothing.
    pub fn insert_chain_from_kv(
        &self,
        model_key: &str,
        request_id: &str,
        kv_store: &KvCacheStore,
        block_tokens: u32,
        keys: &[[u8; 32]],
        // As on the token path: asked once the save knows what it replaces.
        room: impl RoomPlan,
    ) -> u32 {
        let bt = block_tokens as usize;
        if !self.enabled || bt == 0 || keys.is_empty() {
            return 0;
        }
        let key = KvCacheStore::cache_key(model_key, request_id);
        let Ok(held) = HeldCache::read(kv_store, &key) else {
            return 0;
        };
        // A split segment's state is refused: a chain entry carries
        // attention layers only.
        if !held.recurrent_at.is_empty() {
            return 0;
        }
        let ceiling = self.positions_ceiling(held.cost);
        let whole = held.positions.min(keys.len() * bt).min(ceiling) / bt * bt;
        if whole == 0 || whole < self.min_tokens {
            return 0;
        }
        // Already held, whole: nothing to copy, and nothing given up.
        if let Some(blocks) =
            self.covered_by_chain(model_key, &keys[..whole / bt], block_tokens, whole)
        {
            return blocks;
        }
        let serves = held.serves;
        let planned_let_go = self.let_go(model_key, |e| {
            e.replaced_by_chain(&keys[..whole / bt], block_tokens, serves)
        });
        let room = room.plan(planned_let_go);
        let positions = whole.min(room.max_positions) / bt * bt;
        let blocks = positions / bt;
        if blocks == 0 || positions < self.min_tokens {
            return 0;
        }
        // As on the token path: a slid window cannot be saved short of where
        // it still answers (FUTURE_WORK #281).
        if positions < serves {
            tracing::info!(
                model_key,
                positions,
                serves_from = serves,
                "prefix-cache: not storing a split's prompt — the sliding-window layers no \
                 longer hold the opening that fits"
            );
            return 0;
        }
        let chain = &keys[..blocks];
        if positions < whole {
            if let Some(blocks) = self.covered_by_chain(model_key, chain, block_tokens, positions) {
                return blocks;
            }
            let let_go = self.let_go(model_key, |e| {
                e.replaced_by_chain(chain, block_tokens, serves)
            });
            if let_go.in_home(room.home) < planned_let_go.in_home(room.home) {
                tracing::info!(
                    model_key,
                    positions,
                    whole,
                    "prefix-cache: not storing a split's prompt — the room's cut keeps an entry \
                     the plan counted as let go"
                );
                return 0;
            }
        }
        let Some(entry) = kv_store_get(kv_store, &key)
            .filter(|e| HeldCache::positions_of(&e.layers) >= positions)
        else {
            return 0;
        };
        // Room first (#367): the entry of this split that ours extends — the
        // `retain` below — then the caps, then the plan's own eviction.
        let room_made = self.make_room(
            model_key,
            |e| e.replaced_by_chain(chain, block_tokens, serves),
            held.cost.bytes_at(positions),
            room,
        );
        let snap = match snapshot_at(
            &entry.layers,
            &[],
            positions,
            held.dim,
            held.max_seq_len,
            room.home,
        ) {
            Ok(s) => Arc::new(s),
            Err(e) => {
                tracing::warn!(model_key, positions, replaced = room_made.replaced,
                    evicted = room_made.evicted, error = %e,
                    "prefix-cache: storing a split prompt failed — the next turn reads it again");
                return 0;
            }
        };
        drop(entry);
        let Ok(mut inner) = self.inner.write() else {
            return 0;
        };
        let bucket = inner.per_model.entry(model_key.to_string()).or_default();
        // An entry of this split whose chain ours extends is covered by ours —
        // a growing conversation keeps ONE entry, as on the token path, unless
        // it is shorter than where ours answers from.
        let serves = snap.serves_from;
        bucket.retain(|e| !e.replaced_by_chain(chain, block_tokens, serves));
        let tick = self.next_tick();
        bucket.push(Entry {
            tokens: Vec::new(),
            chain: chain.to_vec(),
            chain_block: block_tokens,
            snapshot: snap,
            last_hit: AtomicU64::new(tick),
            // A split segment's entry has no manifest (no tokens), so it is
            // never announced or exported whatever this says; and a request
            // kept here has no segment on another computer to store one for.
            kept_here: false,
        });
        self.evict_over_caps(bucket);
        // Held only if the eviction kept it.
        let kept = bucket
            .iter()
            .any(|e| e.chain_block == block_tokens && e.chain.as_slice() == chain);
        tracing::info!(
            model_key,
            stored_positions = positions,
            kept,
            home = ?room.home,
            replaced_before_copy = room_made.replaced,
            evicted_before_copy = room_made.evicted,
            "DIAG: prefix-cache stored a split's prompt"
        );
        if kept {
            blocks as u32
        } else {
            0
        }
    }

    /// Item 8 Phase 2b: look up a cached snapshot whose chained-hash
    /// manifest contains `block_hash`, serialize it, and return the wire
    /// bytes. Returns `None` when no matching entry exists (eviction race
    /// or never cached for this hash). Used by the serving-side
    /// `ExportPrefixSnapshot` handler.
    ///
    /// We must return a snapshot whose `token_count` equals the block
    /// boundary the hash represents — so we build a fresh snapshot by
    /// narrowing the entry's live KV tensors to that position. The
    /// PrefixCache doesn't store per-block sub-snapshots (would bloat
    /// memory); it stores one snapshot per entry at the entry's full
    /// length, plus the ability to re-narrow to any block boundary at
    /// serve time via the entry's KvSnapshot layers (already
    /// complete-seq-len copies narrowed at insert time).
    pub fn export_snapshot_bytes(&self, model_key: &str, block_hash: &[u8; 32]) -> Option<Vec<u8>> {
        if self.block_tokens == 0 {
            return None;
        }
        let inner = self.inner.read().ok()?;
        let bucket = inner.per_model.get(model_key)?;
        let block_size = self.block_tokens;
        // Find the entry whose manifest contains `block_hash`, along with
        // the block index where it appears (== token boundary).
        for entry in bucket {
            let manifest = entry.manifest(block_size);
            if let Some((i, bm)) = manifest
                .iter()
                .enumerate()
                .find(|(_, bm)| bm.block_hash == *block_hash)
            {
                let target_tokens = bm.token_count as usize;
                // Narrow the stored snapshot to the target boundary if the
                // entry itself is longer — each entry holds a snapshot at
                // `entry.tokens.len()` tokens but we want `target_tokens`.
                let narrowed = narrow_snapshot(&entry.snapshot, target_tokens).ok()?;
                let bytes = serialize_snapshot_with_block_size(
                    &narrowed,
                    &entry.tokens[..target_tokens],
                    Some(block_size),
                )
                .ok()?;
                tracing::debug!(
                    model_key,
                    block_index = i,
                    target_tokens,
                    bytes_len = bytes.len(),
                    "DIAG: export_snapshot_bytes HIT"
                );
                return Some(bytes);
            }
        }
        None
    }

    /// Snapshot the current per-model BLAKE3 block-hash manifest. Used by
    /// tests to validate the cache's announce-payload shape; production
    /// re-announce flow uses `enumerate_manifest_locked` directly from
    /// `insert_from_kv`. Returns an empty vec when the model has no entries
    /// or `block_tokens == 0`.
    #[cfg(test)]
    pub fn enumerate_manifest(&self, model_key: &str) -> Vec<PrefixBlockEntry> {
        if self.block_tokens == 0 {
            return Vec::new();
        }
        let Ok(inner) = self.inner.read() else {
            return Vec::new();
        };
        let Some(bucket) = inner.per_model.get(model_key) else {
            return Vec::new();
        };
        enumerate_manifest_locked(bucket, self.block_tokens)
    }

    /// Item 8 Phase 2b: hydrate a per-request KV entry directly from
    /// serialized snapshot bytes returned by a remote peer. Returns the
    /// number of tokens seeded, or an error if deserialization failed.
    /// Does NOT re-BLAKE3-verify — the daemon's prefix probe
    /// (`daemon::background`, `verify_fetched_snapshot`) has done that.
    pub(crate) fn hydrate_request_from_bytes(
        &self,
        kv_store: &KvCacheStore,
        model_key: &str,
        request_id: &str,
        bytes: &[u8],
        device: &candle_core::Device,
        layouts: &[Option<CacheLayout>],
    ) -> Result<usize, SwarmError> {
        let (snap, _tokens) = deserialize_snapshot(bytes, device)?;
        self.hydrate_request_from_snapshot(kv_store, model_key, request_id, &snap, layouts)
    }

    /// Seed a fresh `KvCacheEntry` for `request_id` from `snapshot`. Returns
    /// the number of tokens seeded. Creates the entry if it doesn't exist.
    ///
    /// `layouts` is how each layer keeps its cache (`SplitModel::cache_layouts`)
    /// — REQUIRED, because a snapshot cannot carry it: each layer's cache is
    /// built as a forward would build it, and a sliding-window layer takes
    /// only what its window reads. Built from a long snapshot as it used to be,
    /// such a layer was allocated the whole prompt — memory every estimate of
    /// its cost says it never holds (FUTURE_WORK #281). A layer with no layout
    /// is built as before, and its first forward conforms it.
    pub(crate) fn hydrate_request_from_snapshot(
        &self,
        kv_store: &KvCacheStore,
        model_key: &str,
        request_id: &str,
        snapshot: &KvSnapshot,
        layouts: &[Option<CacheLayout>],
    ) -> Result<usize, SwarmError> {
        let num_layers = snapshot.layers.len();
        let key = KvCacheStore::cache_key(model_key, request_id);
        let mut entry = kv_store.get_or_create_keyed(&key, num_layers);
        let started = Instant::now();
        for (i, layer_kv) in snapshot.layers.iter().enumerate() {
            let Some((k_src, v_src)) = layer_kv else {
                continue;
            };
            // A layer kept in host RAM goes back to its card first: the
            // cache below allocates on its source's device (#257).
            let moved;
            let (k_src, v_src) = match snapshot.restore_to.get(i).and_then(Option::as_ref) {
                Some(card) => {
                    moved = k_src
                        .to_device(card)
                        .and_then(|k| Ok((k, v_src.to_device(card)?)))
                        .map_err(|e| {
                            SwarmError::Internal(format!(
                                "prefix-cache hydrate layer {i}: copy back to the card failed: {e}"
                            ))
                        })?;
                    (&moved.0, &moved.1)
                }
                None => (k_src, v_src),
            };
            let kv_slot = &mut entry.layers[i];
            // A slid layer's copy starts past 0, and so does its cache — never
            // more positions than it would have held unslid (FUTURE_WORK #281).
            let from = snapshot.layer_held_from(i);
            let layout = layouts.get(i).copied().flatten();
            // A windowed layer keeps what its window reads, as its first slide
            // would leave it: the rest of a long snapshot is never read again.
            let held = k_src.dim(snapshot.dim).map_err(SwarmError::internal)?;
            let kept = layout
                .and_then(|l| l.window)
                .map(super::kv_cache::positions_kept_for_window)
                .filter(|&keep| held > keep);
            let newest;
            let (from, k_src, v_src) = match kept {
                Some(keep) => {
                    let skip = held - keep;
                    newest = (|| -> candle_core::Result<(Tensor, Tensor)> {
                        Ok((
                            k_src.narrow(snapshot.dim, skip, keep)?,
                            v_src.narrow(snapshot.dim, skip, keep)?,
                        ))
                    })()
                    .map_err(SwarmError::internal)?;
                    (from + skip, &newest.0, &newest.1)
                }
                None => (from, k_src, v_src),
            };
            // Reserve from the snapshot's TOKEN COUNT, not its recorded
            // `max_seq_len`. Snapshots cross the network between peers
            // (`export_snapshot_bytes`), so that field may have been written by
            // a build that reserved a model's whole context window up front —
            // honouring it would re-inflate the reservation this node just
            // stopped making. See `KV_CACHE_GROWTH_TOKENS`.
            //
            // And at least what the worker reserved for the whole prompt this
            // snapshot is a prefix of, so the suffix that follows is appended
            // into the same allocation rather than grown into (FUTURE_WORK
            // #32). Growth past that is by the ordinary quantum.
            let mut kv = LayerKv::with_capacity(
                snapshot.dim,
                crate::inference::layers::kv_cache_reservation(snapshot.token_count)
                    .max(kv_store.reserved_positions(request_id))
                    .saturating_sub(from)
                    .max(1),
                crate::inference::layers::KV_CACHE_GROWTH_TOKENS,
            );
            if let Some(layout) = layout {
                kv.conform_to(layout.storage).map_err(|e| {
                    SwarmError::Internal(format!(
                        "prefix-cache hydrate layer {i}: conform failed: {e}"
                    ))
                })?;
            }
            kv.begin_at(from);
            // Told its window before the first append, a windowed cache is
            // born its window and one quantum wide (`LayerKv::slide_beyond`).
            if let Some(layout) = layout {
                kv.slide_beyond(layout.window);
            }
            kv.append(k_src, v_src).map_err(|e| {
                SwarmError::Internal(format!(
                    "prefix-cache hydrate layer {i}: append failed: {e}"
                ))
            })?;
            *kv_slot = Some(kv);
        }
        // Each recurrent layer gets a COPY on its own device: a card's steps
        // write into the pair they read two steps later (`SsmState`), and the
        // snapshot must stay as saved for the next hit (#228).
        for (i, state) in snapshot.ssm.iter().enumerate() {
            let Some((conv, rec)) = state else {
                continue;
            };
            let card = snapshot.restore_to.get(i).and_then(Option::as_ref);
            let copy = |t: &Tensor| match card {
                Some(card) => t.to_device(card),
                None => t.force_contiguous(),
            };
            let slot = copy(conv)
                .and_then(|conv_state| {
                    Ok(SsmSlot {
                        conv_state,
                        recurrent_state: copy(rec)?,
                    })
                })
                .map_err(|e| {
                    SwarmError::Internal(format!(
                        "prefix-cache hydrate layer {i}: recurrent state copy failed: {e}"
                    ))
                })?;
            let Some(held) = entry.ssm_states.get_mut(i) else {
                return Err(SwarmError::Internal(format!(
                    "prefix-cache hydrate: the snapshot's recurrent layer {i} is past the \
                     request's {} layers",
                    num_layers
                )));
            };
            *held = Some(SsmState::new(slot, snapshot.token_count));
        }
        entry.last_accessed = Instant::now();
        if snapshot.home() == SnapshotHome::HostRam {
            tracing::info!(
                model_key,
                request_id,
                positions = snapshot.token_count,
                mb = snapshot.bytes() / (1024 * 1024),
                copy_ms = started.elapsed().as_millis() as u64,
                "DIAG: prefix-cache restored a snapshot from host RAM to the card"
            );
        }
        Ok(snapshot.token_count)
    }

    #[cfg(test)]
    pub fn entry_count(&self, model_key: &str) -> usize {
        self.inner
            .read()
            .ok()
            .and_then(|i| i.per_model.get(model_key).map(|v| v.len()))
            .unwrap_or(0)
    }

    /// Bytes of KV held in `home` across every model. `BesideLive` is what
    /// this cache costs the device right now, charged against the KV budget
    /// by `admit_prompt`; `HostRam` is what it holds off the card (#257),
    /// which that budget never sees.
    pub fn bytes_in(&self, home: SnapshotHome) -> usize {
        self.inner
            .read()
            .ok()
            .map(|i| {
                i.per_model
                    .values()
                    .flatten()
                    .map(|e| e.snapshot.bytes_in(home))
                    .sum()
            })
            .unwrap_or(0)
    }

    /// Evict least-recently-hit entries kept in `home`, across every model,
    /// until at least `needed` bytes of that memory are freed or none is left.
    /// Returns the bytes freed.
    ///
    /// Called when a request's own KV would not fit beside the cache (gotcha
    /// #440). A cached prompt is reconstructible; the request in hand is not,
    /// and a cache that pushes live memory onto the host costs far more than
    /// the prefill it saves. Oldest first, so an entry the current request
    /// just hit — whose `last_hit` the lookup bumped — goes last. **Only
    /// entries in `home` go**: dropping a snapshot kept in host RAM frees
    /// nothing on the card, and counting it as freed would let a refusal the
    /// card still owes through (#257).
    pub fn release(&self, needed: usize, home: SnapshotHome) -> usize {
        let Ok(mut inner) = self.inner.write() else {
            return 0;
        };
        let gone = take_least_recently_hit(&mut inner, needed as u64, home);
        let freed: usize = gone.iter().map(|e| e.snapshot.bytes()).sum();
        if freed > 0 {
            tracing::info!(
                freed_mb = freed / (1024 * 1024),
                needed_mb = needed / (1024 * 1024),
                home = ?home,
                remaining_mb = inner
                    .per_model
                    .values()
                    .flatten()
                    .map(|e| e.snapshot.bytes_in(home))
                    .sum::<usize>()
                    / (1024 * 1024),
                "DIAG: prefix-cache released cached prompts to make room"
            );
        }
        // The entries go once the lock is released: freeing 0.5 GB is not
        // instant.
        drop(inner);
        drop(gone);
        freed
    }

    /// Bytes of KV currently held for a model, wherever it is kept. The
    /// figure the byte budget bounds, and the one a count of entries cannot
    /// express.
    #[cfg(test)]
    pub fn bytes_held(&self, model_key: &str) -> usize {
        self.inner
            .read()
            .ok()
            .and_then(|i| {
                i.per_model
                    .get(model_key)
                    .map(|b| b.iter().map(|e| e.snapshot.bytes()).sum())
            })
            .unwrap_or(0)
    }
}

/// What a request's cache holds, read once for a save and let go before the
/// save's room is planned: the plan reads the store
/// (`KvCacheStore::occupancy`), never under a guard on one of its entries.
struct HeldCache {
    dim: usize,
    max_seq_len: usize,
    /// Positions the cache holds (its first populated layer's length).
    positions: usize,
    cost: SnapshotCost,
    /// Where a snapshot of it answers from ([`serves_from`]).
    serves: usize,
    /// The position each recurrent layer's state is at (#228); empty for a
    /// model without one.
    recurrent_at: Vec<usize>,
}

impl HeldCache {
    fn read(kv_store: &KvCacheStore, key: &str) -> Result<Self, &'static str> {
        let entry = kv_store_get(kv_store, key).ok_or("no KV entry to snapshot")?;
        let first_kv = entry
            .layers
            .iter()
            .flatten()
            .next()
            .ok_or("the KV entry has no populated layers")?;
        Ok(Self {
            dim: first_kv.k_cache().dim(),
            max_seq_len: first_kv.k_cache().max_seq_len(),
            positions: first_kv.current_seq_len(),
            cost: SnapshotCost::of(&entry.layers),
            serves: serves_from(&entry.layers),
            recurrent_at: entry
                .ssm_states
                .iter()
                .flatten()
                .map(SsmState::positions)
                .collect(),
        })
    }

    /// Positions `layers` hold, as [`Self::positions`] reads them.
    fn positions_of(layers: &[Option<LayerKv>]) -> usize {
        layers
            .iter()
            .flatten()
            .next()
            .map_or(0, |kv| kv.current_seq_len())
    }

    /// Where a snapshot taken at `at` answers from (`snapshot_at`'s figure):
    /// a recurrent state's serves only its whole length.
    fn serves_once_saved(&self, at: usize) -> usize {
        if self.recurrent_at.is_empty() {
            self.serves
        } else {
            at
        }
    }
}

/// Takes the least recently hit entries kept in `home`, across every model,
/// out of `inner` until at least `needed` bytes of that memory are gone or
/// none is left — the ONE eviction order of [`PrefixCache::release`] and of a
/// save's planned room ([`PrefixCache::make_room`]). The caller drops them
/// once its lock is released.
fn take_least_recently_hit(inner: &mut Inner, needed: u64, home: SnapshotHome) -> Vec<Entry> {
    let mut gone = Vec::new();
    let mut freed = 0u64;
    while freed < needed {
        let oldest = inner
            .per_model
            .iter()
            .flat_map(|(key, bucket)| {
                bucket
                    .iter()
                    .enumerate()
                    .filter(|(_, e)| e.snapshot.home() == home)
                    .map(move |(i, e)| (e.last_hit.load(Ordering::Relaxed), key.clone(), i))
            })
            .min_by_key(|(hit, _, _)| *hit);
        let Some((_, key, idx)) = oldest else {
            break;
        };
        let bucket = inner
            .per_model
            .get_mut(&key)
            .expect("key came from the map");
        let removed = bucket.remove(idx);
        if bucket.is_empty() {
            inner.per_model.remove(&key);
        }
        freed = freed.saturating_add(removed.snapshot.bytes() as u64);
        gone.push(removed);
    }
    gone
}

fn kv_store_get<'a>(
    kv_store: &'a KvCacheStore,
    key: &str,
) -> Option<dashmap::mapref::one::Ref<'a, String, KvCacheEntry>> {
    kv_store.get_entry(key)
}

/// Build the deduped union of every entry's chained block manifest in a
/// bucket. Two entries in the same bucket whose tokens share a common prefix
/// produce identical block hashes for the shared portion (chained-hash
/// prefix property), so dedup-by-`block_hash` is correct and avoids
/// re-broadcasting the same block under multiple entries.
fn enumerate_manifest_locked(bucket: &[Entry], block_size: usize) -> Vec<PrefixBlockEntry> {
    if block_size == 0 || bucket.is_empty() {
        return Vec::new();
    }
    let mut seen: std::collections::HashSet<[u8; 32]> = std::collections::HashSet::new();
    let mut out = Vec::new();
    for entry in bucket {
        for block in entry.manifest(block_size) {
            if seen.insert(block.block_hash) {
                out.push(block);
            }
        }
    }
    out
}

/// Narrow an existing `KvSnapshot` to `target_tokens` on the seq dim.
/// Returns a fresh snapshot with independent tensors (contiguous copies),
/// so the caller can safely serialize + ship it without aliasing the
/// source. `target_tokens` must be `<= snap.token_count`.
fn narrow_snapshot(snap: &KvSnapshot, target_tokens: usize) -> Result<KvSnapshot, SwarmError> {
    if target_tokens > snap.token_count {
        return Err(SwarmError::Internal(format!(
            "narrow_snapshot: target {} > current {}",
            target_tokens, snap.token_count
        )));
    }
    if target_tokens < snap.serves_from {
        return Err(SwarmError::Internal(format!(
            "narrow_snapshot: target {target_tokens} is before the {} a sliding-window layer \
             can continue from (FUTURE_WORK #281)",
            snap.serves_from
        )));
    }
    if target_tokens == snap.token_count {
        // Shallow clone — candle Tensors are ref-counted.
        let layers: Vec<Option<(Tensor, Tensor)>> = snap
            .layers
            .iter()
            .map(|kv| kv.as_ref().map(|(k, v)| (k.clone(), v.clone())))
            .collect();
        return Ok(KvSnapshot {
            token_count: target_tokens,
            layers,
            dim: snap.dim,
            max_seq_len: snap.max_seq_len,
            restore_to: snap.restore_to.clone(),
            held_from: snap.held_from.clone(),
            serves_from: snap.serves_from,
            ssm: snap.ssm.clone(),
        });
    }
    // Unreachable for one today — its `serves_from` is its whole length, so
    // only the whole-length clone above passes — but a recurrent state cannot
    // be cut back, and a new caller must not get a snapshot without it.
    if snap.carries_recurrent_state() {
        return Err(SwarmError::Internal(format!(
            "narrow_snapshot: a recurrent state of {} positions cannot be cut back to \
             {target_tokens} (#228)",
            snap.token_count
        )));
    }
    let mut out: Vec<Option<(Tensor, Tensor)>> = Vec::with_capacity(snap.layers.len());
    for (i, kv_opt) in snap.layers.iter().enumerate() {
        let Some((k, v)) = kv_opt else {
            out.push(None);
            continue;
        };
        // A layer kept in host RAM stays a VIEW: compacting it here is a
        // fresh host copy of the whole layer (a hit cost 0.2-0.9 s more than
        // on the card for a 4.8K-token 7B on WSL2, 2026-10-13), while the
        // restore copies the layer to its card as it is and
        // `LayerKv::append` compacts it there (#257). Every other reader of a
        // narrowed snapshot (`serialize_snapshot_*`, `snapshot_is_finite`)
        // makes its own contiguous copy.
        let stays_a_view = snap.restore_to.get(i).is_some_and(Option::is_some);
        // A slid layer's copy starts past 0; `serves_from` kept the target past it.
        let held = target_tokens - snap.layer_held_from(i);
        let narrow =
            |t: &Tensor| {
                t.narrow(snap.dim, 0, held).and_then(|n| {
                    if stays_a_view {
                        Ok(n)
                    } else {
                        n.contiguous()
                    }
                })
            };
        let k_narrow =
            narrow(k).map_err(|e| SwarmError::Internal(format!("narrow_snapshot k: {e}")))?;
        let v_narrow =
            narrow(v).map_err(|e| SwarmError::Internal(format!("narrow_snapshot v: {e}")))?;
        out.push(Some((k_narrow, v_narrow)));
    }
    Ok(KvSnapshot {
        token_count: target_tokens,
        layers: out,
        dim: snap.dim,
        max_seq_len: snap.max_seq_len,
        restore_to: snap.restore_to.clone(),
        held_from: snap.held_from.clone(),
        serves_from: snap.serves_from,
        ssm: Vec::new(),
    })
}

/// The shortest prefix a snapshot of `layers` could be restored at —
/// [`LayerKv::serves_from`] over every layer, 0 when none has slid.
fn serves_from(layers: &[Option<LayerKv>]) -> usize {
    layers
        .iter()
        .flatten()
        .map(LayerKv::serves_from)
        .max()
        .unwrap_or(0)
}

/// Copy positions `0..pos` of every layer's cache into a snapshot kept in
/// `home` — from where a sliding-window layer's slid cache starts, for that
/// layer (FUTURE_WORK #281). For `HostRam` each card layer is narrowed on the
/// card, copied to host memory and its card copy dropped before the next
/// layer's, so the card never holds more than one layer's transient (~4 MB on
/// a 7B at 4K tokens).
///
/// A recurrent layer's state (`ssm_states`, by layer) is copied too, and must
/// describe exactly `pos` positions; such a snapshot serves only its whole
/// length (`KvSnapshot::serves_from`, #228).
fn snapshot_at(
    layers: &[Option<LayerKv>],
    ssm_states: &[Option<SsmState>],
    pos: usize,
    dim: usize,
    max_seq_len: usize,
    home: SnapshotHome,
) -> Result<KvSnapshot, SwarmError> {
    let mut serves_from = serves_from(layers);
    if pos < serves_from {
        return Err(SwarmError::Internal(format!(
            "snapshot_at: position {pos} is before the {serves_from} a sliding-window layer \
             can continue from (FUTURE_WORK #281)"
        )));
    }
    let mut out: Vec<Option<(Tensor, Tensor)>> = Vec::with_capacity(layers.len());
    let mut restore_to: Vec<Option<Device>> = Vec::with_capacity(layers.len());
    let mut held_from: Vec<usize> = Vec::with_capacity(layers.len());
    let mut ssm: Vec<Option<(Tensor, Tensor)>> = Vec::new();
    // A COPY kept in `home`: on a card the live pair is written into in place
    // two steps later (`SsmState`), and `force_contiguous` always copies.
    let keep = |t: &Tensor| -> candle_core::Result<(Tensor, Option<Device>)> {
        let copy = t.force_contiguous()?;
        let card = copy.device().clone();
        if home == SnapshotHome::HostRam && !card.is_cpu() {
            Ok((copy.to_device(&Device::Cpu)?, Some(card)))
        } else {
            Ok((copy, None))
        }
    };
    for (i, kv_opt) in layers.iter().enumerate() {
        let Some(kv) = kv_opt else {
            let state = ssm_states.get(i).and_then(Option::as_ref);
            let Some(state) = state else {
                out.push(None);
                restore_to.push(None);
                held_from.push(0);
                continue;
            };
            if state.positions() != pos {
                return Err(SwarmError::Internal(format!(
                    "snapshot_at: layer {i}'s recurrent state is at {} positions, not {pos}",
                    state.positions()
                )));
            }
            let slot = state.current();
            let ((conv, card), (rec, _)) = keep(&slot.conv_state)
                .and_then(|c| Ok((c, keep(&slot.recurrent_state)?)))
                .map_err(|e| SwarmError::Internal(format!("snapshot recurrent state: {e}")))?;
            ssm.resize_with(i, || None);
            ssm.push(Some((conv, rec)));
            out.push(None);
            restore_to.push(card);
            held_from.push(0);
            serves_from = pos;
            continue;
        };
        let cur = kv.current_seq_len();
        if cur < pos {
            return Err(SwarmError::Internal(format!(
                "snapshot_at: layer seq_len {cur} < requested pos {pos}"
            )));
        }
        let from = kv.held_from();
        held_from.push(from);
        let k_src = kv
            .k()
            .map_err(|e| SwarmError::Internal(format!("snapshot_at: k() failed: {e}")))?
            .ok_or_else(|| SwarmError::Internal("snapshot_at: k() returned None".into()))?;
        let v_src = kv
            .v()
            .map_err(|e| SwarmError::Internal(format!("snapshot_at: v() failed: {e}")))?
            .ok_or_else(|| SwarmError::Internal("snapshot_at: v() returned None".into()))?;
        // Narrow to [from..pos] on seq dim and COPY, so the snapshot never
        // aliases the live buffer. `contiguous()` is not a copy: it hands back
        // the same storage for a view candle already counts as contiguous, and
        // a one-key/value-head cache narrowed along its positions is one
        // (`is_contiguous` skips size-1 dims) — Gemma 3 1B's snapshot shared
        // its live cache, and a window's in-place slide rewrote it.
        let k_snap = k_src
            .narrow(dim, 0, pos - from)
            .and_then(|t| t.force_contiguous())
            .map_err(|e| SwarmError::Internal(format!("snapshot k narrow: {e}")))?;
        let v_snap = v_src
            .narrow(dim, 0, pos - from)
            .and_then(|t| t.force_contiguous())
            .map_err(|e| SwarmError::Internal(format!("snapshot v narrow: {e}")))?;
        let card = k_snap.device().clone();
        if home == SnapshotHome::HostRam && !card.is_cpu() {
            let k_host = k_snap
                .to_device(&Device::Cpu)
                .map_err(|e| SwarmError::Internal(format!("snapshot k to host: {e}")))?;
            let v_host = v_snap
                .to_device(&Device::Cpu)
                .map_err(|e| SwarmError::Internal(format!("snapshot v to host: {e}")))?;
            out.push(Some((k_host, v_host)));
            restore_to.push(Some(card));
        } else {
            out.push(Some((k_snap, v_snap)));
            restore_to.push(None);
        }
    }
    Ok(KvSnapshot {
        token_count: pos,
        layers: out,
        dim,
        max_seq_len,
        restore_to,
        held_from,
        serves_from,
        ssm,
    })
}

// ---- Item 8 Phase 2: KvSnapshot wire serialization -------------------------
//
// Format (all lengths little-endian):
//
//   [0..4]    magic = KV_SNAPSHOT_MAGIC ("SKVX")
//   [4..8]    version = KV_SNAPSHOT_VERSION (u32)
//   [8..16]   header_len (u64)
//   [16..16+H] serde_json header (see `SnapshotHeader`)
//   [16+H..]  concatenated per-layer [K f32 LE | V f32 LE] bytes, in the
//             same order as `header.layers`. Only layers with `Some(meta)`
//             contribute bytes; `None` entries emit zero bytes.
//
// Every K/V tensor is cast to `DType::F32` on the sender side before
// encoding. The receiver casts back to whatever its local `KvCache`
// expects (usually f32 on CPU, f16 on CUDA). This costs 2× wire size for
// an f16 sender but is portable across GPU/CPU peer pairings — a common
// configuration on SwarmLLM's heterogeneous swarm. Compression is
// handled one level up by the request_response zstd wrapper.

#[derive(serde::Serialize, serde::Deserialize)]
struct SnapshotLayerMeta {
    k_shape: Vec<usize>,
    v_shape: Vec<usize>,
    /// Number of f32 values in K (same count for V). Used to offset the
    /// binary body on receive.
    k_count: usize,
}

#[derive(serde::Serialize, serde::Deserialize)]
struct SnapshotHeader {
    token_count: usize,
    /// Full u32 token IDs covering the snapshot. Receiver uses these to
    /// BLAKE3-verify the returned block matches the hash they requested.
    tokens: Vec<u32>,
    dim: usize,
    max_seq_len: usize,
    /// Block size the sender used to chain-hash the prompt prefix. The
    /// receiver re-hashes `tokens[..token_count]` at this `block_size` to
    /// verify against the requested `block_hash`. Optional for backward
    /// compatibility with pre-Phase-3 snapshots — when absent, the
    /// verifier falls back to a list of common block sizes.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    block_size: Option<usize>,
    /// Per-layer metadata; `None` marks layers the sender had no cache for
    /// (shouldn't happen for a completed prefill, tolerated for robustness).
    layers: Vec<Option<SnapshotLayerMeta>>,
}

/// Serialize a `KvSnapshot` (plus the token IDs it covers) to a wire frame.
///
/// `tokens.len()` must be `>= snap.token_count`; only `tokens[..token_count]`
/// is recorded in the header. Every tensor in `snap.layers` is cast to f32
/// before encoding so the frame is device-independent.
///
/// `block_size: Some(N)` records the sender's block size so the receiver can
/// BLAKE3-verify `tokens[..token_count]` against the requested block hash
/// deterministically — every production caller passes it. `None` is tolerated
/// for pre-Phase-3 compatibility, where the verifier falls back to guessing
/// common defaults; the round-trip tests still cover that path.
pub fn serialize_snapshot_with_block_size(
    snap: &KvSnapshot,
    tokens: &[u32],
    block_size: Option<usize>,
) -> Result<Vec<u8>, SwarmError> {
    if tokens.len() < snap.token_count {
        return Err(SwarmError::Internal(format!(
            "serialize_snapshot: tokens.len() {} < token_count {}",
            tokens.len(),
            snap.token_count
        )));
    }
    // The format describes every layer from position 0; a reader would take a
    // slid layer's copy as the opening (FUTURE_WORK #281). `Entry::manifest`
    // never offers one — this keeps a new caller from sending it anyway.
    if !snap.holds_every_position() {
        return Err(SwarmError::Internal(
            "serialize_snapshot: a sliding-window layer of this snapshot no longer holds \
             the prompt's opening"
                .into(),
        ));
    }
    // Nor does it carry a recurrent state (#228): the attention layers alone
    // would restore beside a state that starts over.
    if snap.carries_recurrent_state() {
        return Err(SwarmError::Internal(
            "serialize_snapshot: this snapshot carries a recurrent state, which the wire \
             format does not"
                .into(),
        ));
    }
    let mut layer_meta: Vec<Option<SnapshotLayerMeta>> = Vec::with_capacity(snap.layers.len());
    let mut body: Vec<u8> = Vec::new();
    for kv_opt in &snap.layers {
        let Some((k, v)) = kv_opt else {
            layer_meta.push(None);
            continue;
        };
        let k_f32 = k
            .to_dtype(candle_core::DType::F32)
            .and_then(|t| t.contiguous())
            .map_err(|e| SwarmError::Internal(format!("serialize_snapshot k cast: {e}")))?;
        let v_f32 = v
            .to_dtype(candle_core::DType::F32)
            .and_then(|t| t.contiguous())
            .map_err(|e| SwarmError::Internal(format!("serialize_snapshot v cast: {e}")))?;
        let k_shape: Vec<usize> = k_f32.shape().dims().to_vec();
        let v_shape: Vec<usize> = v_f32.shape().dims().to_vec();
        let k_vec: Vec<f32> = k_f32
            .flatten_all()
            .and_then(|t| t.to_vec1::<f32>())
            .map_err(|e| SwarmError::Internal(format!("serialize_snapshot k flatten: {e}")))?;
        let v_vec: Vec<f32> = v_f32
            .flatten_all()
            .and_then(|t| t.to_vec1::<f32>())
            .map_err(|e| SwarmError::Internal(format!("serialize_snapshot v flatten: {e}")))?;
        let k_count = k_vec.len();
        body.reserve(4 * (k_vec.len() + v_vec.len()));
        for f in &k_vec {
            body.extend_from_slice(&f.to_le_bytes());
        }
        for f in &v_vec {
            body.extend_from_slice(&f.to_le_bytes());
        }
        layer_meta.push(Some(SnapshotLayerMeta {
            k_shape,
            v_shape,
            k_count,
        }));
    }
    let header = SnapshotHeader {
        token_count: snap.token_count,
        tokens: tokens[..snap.token_count].to_vec(),
        dim: snap.dim,
        max_seq_len: snap.max_seq_len,
        block_size,
        layers: layer_meta,
    };
    let header_bytes =
        serde_json::to_vec(&header).map_err(|e| SwarmError::Internal(format!("header: {e}")))?;
    let header_len = header_bytes.len() as u64;
    let mut out = Vec::with_capacity(4 + 4 + 8 + header_bytes.len() + body.len());
    out.extend_from_slice(KV_SNAPSHOT_MAGIC);
    out.extend_from_slice(&KV_SNAPSHOT_VERSION.to_le_bytes());
    out.extend_from_slice(&header_len.to_le_bytes());
    out.extend_from_slice(&header_bytes);
    out.extend_from_slice(&body);
    Ok(out)
}

/// Item 8 Phase 3: sanity-check a just-deserialized KV snapshot for
/// numerical corruption. Returns `true` if every populated layer's K and
/// V tensors contain only finite values (no NaN, no ±Inf). A failure
/// here almost always indicates a malicious or broken peer — the caller
/// drops the snapshot and penalizes trust. Called BEFORE hydration so a
/// bad peer can never poison our KV cache.
pub fn snapshot_is_finite(snap: &KvSnapshot) -> bool {
    for kv_opt in &snap.layers {
        let Some((k, v)) = kv_opt else { continue };
        for tensor in [k, v] {
            let flat: Vec<f32> = match tensor
                .to_dtype(candle_core::DType::F32)
                .and_then(|t| t.flatten_all())
                .and_then(|t| t.to_vec1::<f32>())
            {
                Ok(v) => v,
                Err(_) => return false,
            };
            if flat.iter().any(|f| !f.is_finite()) {
                return false;
            }
        }
    }
    true
}

/// Deserialize a wire-format KV snapshot on `device`. Returns the
/// reconstructed `KvSnapshot` plus the token IDs the sender claimed the
/// snapshot covers (caller MUST re-hash these and match against the
/// requested block hash before trusting the data).
pub fn deserialize_snapshot(
    bytes: &[u8],
    device: &candle_core::Device,
) -> Result<(KvSnapshot, Vec<u32>), SwarmError> {
    deserialize_snapshot_full(bytes, device).map(|(s, t, _)| (s, t))
}

/// Phase 3 variant: also returns the sender's `block_size` from the
/// header (`None` for pre-Phase-3 senders that didn't record it).
pub fn deserialize_snapshot_full(
    bytes: &[u8],
    device: &candle_core::Device,
) -> Result<(KvSnapshot, Vec<u32>, Option<usize>), SwarmError> {
    // Header framing: magic(4) + version(4) + header_len(8) = 16 bytes.
    if bytes.len() < 16 {
        return Err(SwarmError::Internal("snapshot: frame too short".into()));
    }
    if &bytes[..4] != KV_SNAPSHOT_MAGIC {
        return Err(SwarmError::Internal("snapshot: magic mismatch".into()));
    }
    let version = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
    if version != KV_SNAPSHOT_VERSION {
        return Err(SwarmError::Internal(format!(
            "snapshot: unsupported version {version}"
        )));
    }
    // Every size below is the SENDER's (a peer's snapshot reaches here through
    // `verify_fetched_snapshot`), so each sum and product is checked: wrapped,
    // `16 + header_len` passed the length test and the slice panicked the probe
    // task (sweep 2026-10-24).
    let header_end = usize::try_from(u64::from_le_bytes(bytes[8..16].try_into().unwrap()))
        .ok()
        .and_then(|len| len.checked_add(16))
        .filter(|&end| end <= bytes.len())
        .ok_or_else(|| SwarmError::Internal("snapshot: truncated header".into()))?;
    let header: SnapshotHeader = serde_json::from_slice(&bytes[16..header_end])
        .map_err(|e| SwarmError::Internal(format!("snapshot header decode: {e}")))?;
    let body = &bytes[header_end..];
    let mut cursor = 0usize;
    let mut layers: Vec<Option<(Tensor, Tensor)>> = Vec::with_capacity(header.layers.len());
    for meta_opt in header.layers {
        let Some(meta) = meta_opt else {
            layers.push(None);
            continue;
        };
        // K's shape must account for exactly the values sent: candle's own
        // check multiplies the dims unchecked, so a shape whose product wraps
        // to `k_count` built a tensor far larger than its storage.
        let product = |dims: &[usize]| dims.iter().try_fold(1usize, |n, &d| n.checked_mul(d));
        let sizes = product(&meta.v_shape)
            .filter(|_| product(&meta.k_shape) == Some(meta.k_count))
            .and_then(|v_count| {
                let k_bytes = meta.k_count.checked_mul(4)?;
                let v_bytes = v_count.checked_mul(4)?;
                let end = cursor.checked_add(k_bytes)?.checked_add(v_bytes)?;
                (end <= body.len()).then_some((v_count, k_bytes, v_bytes))
            });
        let Some((v_count, k_bytes, v_bytes)) = sizes else {
            return Err(SwarmError::Internal(format!(
                "snapshot: body too short at layer (have {} bytes)",
                body.len() - cursor
            )));
        };
        let k_vec: Vec<f32> = (0..meta.k_count)
            .map(|i| {
                let off = cursor + i * 4;
                f32::from_le_bytes([body[off], body[off + 1], body[off + 2], body[off + 3]])
            })
            .collect();
        cursor += k_bytes;
        let v_vec: Vec<f32> = (0..v_count)
            .map(|i| {
                let off = cursor + i * 4;
                f32::from_le_bytes([body[off], body[off + 1], body[off + 2], body[off + 3]])
            })
            .collect();
        cursor += v_bytes;
        let k_tensor = Tensor::from_vec(k_vec, meta.k_shape.as_slice(), device)
            .map_err(|e| SwarmError::Internal(format!("snapshot k rebuild: {e}")))?;
        let v_tensor = Tensor::from_vec(v_vec, meta.v_shape.as_slice(), device)
            .map_err(|e| SwarmError::Internal(format!("snapshot v rebuild: {e}")))?;
        layers.push(Some((k_tensor, v_tensor)));
    }
    let snap = KvSnapshot {
        token_count: header.token_count,
        layers,
        dim: header.dim,
        max_seq_len: header.max_seq_len,
        restore_to: Vec::new(),
        held_from: Vec::new(),
        serves_from: 0,
        ssm: Vec::new(),
    };
    Ok((snap, header.tokens, header.block_size))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    /// Room for any snapshot, kept beside the live cache — every test here
    /// runs on the processor, where that is the only home.
    const ANYWHERE: SnapshotRoom = SnapshotRoom {
        home: SnapshotHome::BesideLive,
        max_positions: usize::MAX,
        evict_bytes: 0,
    };

    fn beside(max_positions: usize) -> SnapshotRoom {
        SnapshotRoom {
            home: SnapshotHome::BesideLive,
            max_positions,
            evict_bytes: 0,
        }
    }

    fn make_fake_kv(
        kv_store: &KvCacheStore,
        model_key: &str,
        request_id: &str,
        num_layers: usize,
        seq_len: usize,
    ) {
        // Build tiny KV tensors [1, 1, seq_len, 4] on CPU so snapshot math has
        // something concrete to narrow.
        let device = Device::Cpu;
        let mut entry = kv_store.get_or_create(model_key, request_id, num_layers);
        for slot in entry.layers.iter_mut() {
            let k = Tensor::zeros((1usize, 1, seq_len, 4), DType::F32, &device).unwrap();
            let v = Tensor::zeros((1usize, 1, seq_len, 4), DType::F32, &device).unwrap();
            // dim=2 is the sequence dim for this shape
            let mut kv = LayerKv::with_capacity(2, 4096, 4096);
            kv.append(&k, &v).unwrap();
            *slot = Some(kv);
        }
    }

    /// Cached prompts are the one thing eviction can recover for a request
    /// that would not otherwise fit on the device (gotcha #440): oldest hit
    /// first, stop once enough is freed, and report what went.
    #[test]
    fn release_evicts_the_least_recently_hit_prompts_until_enough_is_freed() {
        let pc = PrefixCache::new(true, 8, 0, 4, 8192, 0);
        let kv_store = KvCacheStore::new(std::time::Duration::from_secs(600));
        // Three entries of 10 tokens, inserted a, b, c; then b is hit, so the
        // LRU order becomes a, c, b.
        for (req, base) in [("req-a", 100u32), ("req-b", 200), ("req-c", 300)] {
            make_fake_kv(&kv_store, "m", req, 2, 10);
            let tokens: Vec<u32> = (base..base + 10).collect();
            pc.insert_from_kv("m", req, &kv_store, &tokens, ANYWHERE, KeepAsked::Unasked);
        }
        let b: Vec<u32> = (200..210).collect();
        assert!(pc.lookup("m", &b).is_some());
        let per_entry = pc.bytes_in(SnapshotHome::BesideLive) / 3;
        assert!(per_entry > 0);

        // Ask for a little more than one entry: two must go, a then c.
        let freed = pc.release(per_entry + 1, SnapshotHome::BesideLive);
        assert_eq!(freed, 2 * per_entry);
        assert_eq!(pc.bytes_in(SnapshotHome::BesideLive), per_entry);
        let a: Vec<u32> = (100..110).collect();
        let c: Vec<u32> = (300..310).collect();
        assert!(pc.lookup("m", &a).is_none(), "a was the oldest");
        assert!(pc.lookup("m", &c).is_none(), "c was next");
        assert!(pc.lookup("m", &b).is_some(), "b was hit last and survives");

        // Asking for more than everything frees everything and stops.
        assert_eq!(pc.release(usize::MAX, SnapshotHome::BesideLive), per_entry);
        assert_eq!(pc.bytes_in(SnapshotHome::BesideLive), 0);
        assert_eq!(pc.release(1, SnapshotHome::BesideLive), 0);
    }

    /// Mark the stored entry opening with `first_token` as `snapshot_at`
    /// leaves a card's layers in host RAM — the processor standing in for the
    /// card a CPU test cannot open.
    fn move_off_card(pc: &PrefixCache, model_key: &str, first_token: u32) {
        let mut inner = pc.inner.write().unwrap();
        let entry = inner
            .per_model
            .get_mut(model_key)
            .unwrap()
            .iter_mut()
            .find(|e| e.tokens.first() == Some(&first_token))
            .unwrap();
        let s = &entry.snapshot;
        entry.snapshot = Arc::new(KvSnapshot {
            token_count: s.token_count,
            layers: s.layers.clone(),
            dim: s.dim,
            max_seq_len: s.max_seq_len,
            restore_to: vec![Some(Device::Cpu); s.layers.len()],
            held_from: s.held_from.clone(),
            serves_from: s.serves_from,
            ssm: Vec::new(),
        });
    }

    /// #257: a card's snapshot kept in host RAM frees nothing on the card, so
    /// the card's budget is never charged for it and never evicts it — even
    /// when it is the oldest entry held.
    #[test]
    fn a_snapshot_in_host_ram_is_neither_charged_to_nor_evicted_for_the_card() {
        let pc = PrefixCache::new(true, 8, 0, 4, 8192, 0);
        let kv_store = KvCacheStore::new(std::time::Duration::from_secs(600));
        for (req, base) in [("req-a", 100u32), ("req-b", 200)] {
            make_fake_kv(&kv_store, "m", req, 2, 10);
            let tokens: Vec<u32> = (base..base + 10).collect();
            pc.insert_from_kv("m", req, &kv_store, &tokens, ANYWHERE, KeepAsked::Unasked);
        }
        let per_entry = pc.bytes_in(SnapshotHome::BesideLive) / 2;
        assert!(per_entry > 0);
        move_off_card(&pc, "m", 100);
        assert_eq!(pc.bytes_in(SnapshotHome::BesideLive), per_entry);
        assert_eq!(pc.bytes_in(SnapshotHome::HostRam), per_entry);

        // The card asks for room: only the entry on the card goes, though the
        // one in host RAM is older.
        assert_eq!(pc.release(1, SnapshotHome::BesideLive), per_entry);
        assert_eq!(pc.release(1, SnapshotHome::BesideLive), 0);
        let a: Vec<u32> = (100..110).collect();
        let b: Vec<u32> = (200..210).collect();
        assert!(pc.lookup("m", &b).is_none());
        assert!(
            pc.lookup("m", &a).is_some(),
            "host RAM gave the card nothing"
        );

        // Host RAM's own plan evicts it.
        assert_eq!(pc.release(1, SnapshotHome::HostRam), per_entry);
        assert_eq!(pc.bytes_in(SnapshotHome::HostRam), 0);
    }

    /// #257: a hit on a snapshot kept in host RAM — narrowed, as every hit
    /// is — still knows where its layers go back to, and restores them.
    #[test]
    fn a_snapshot_in_host_ram_is_restored_where_its_layers_run() {
        let pc = PrefixCache::new(true, 8, 0, 4, 8192, 0);
        let kv_store = KvCacheStore::new(std::time::Duration::from_secs(600));
        make_fake_kv(&kv_store, "m", "req-a", 2, 10);
        let tokens: Vec<u32> = (100..110).collect();
        pc.insert_from_kv(
            "m",
            "req-a",
            &kv_store,
            &tokens,
            ANYWHERE,
            KeepAsked::Unasked,
        );
        move_off_card(&pc, "m", 100);

        let snap = pc.lookup("m", &tokens).expect("hit");
        assert_eq!(snap.token_count, 9, "an identical prompt hits at len - 1");
        assert_eq!(
            snap.home(),
            SnapshotHome::HostRam,
            "narrowing kept the way back"
        );
        let restored = pc
            .hydrate_request_from_snapshot(&kv_store, "m", "req-b", &snap, &[])
            .unwrap();
        assert_eq!(restored, 9);
        let key = KvCacheStore::cache_key("m", "req-b");
        let entry = kv_store.get_entry(&key).unwrap();
        for kv in entry.layers.iter() {
            assert_eq!(kv.as_ref().unwrap().current_seq_len(), 9);
        }
    }

    /// #257: a hit on a snapshot kept in host RAM narrows each layer to a
    /// VIEW — compacting it on the host was a fresh copy of the whole layer —
    /// and the restore still holds exactly the narrowed prefix. One kept
    /// beside the live cache is compacted where it is, as before.
    #[test]
    fn a_host_ram_snapshot_narrows_to_a_view_and_restores_only_the_prefix() {
        let dev = Device::Cpu;
        // Two heads, so a prefix of the sequence dim is not contiguous.
        let k = Tensor::arange(0f32, 48.0, &dev)
            .unwrap()
            .reshape((1usize, 2, 6, 4))
            .unwrap();
        let v = (&k + 1000.0).unwrap();
        let snap = |restore_to: Vec<Option<Device>>| KvSnapshot {
            token_count: 6,
            layers: vec![Some((k.clone(), v.clone()))],
            dim: 2,
            max_seq_len: 4096,
            restore_to,
            held_from: Vec::new(),
            serves_from: 0,
            ssm: Vec::new(),
        };
        let host = narrow_snapshot(&snap(vec![Some(Device::Cpu)]), 4).unwrap();
        let (hk, hv) = host.layers[0].as_ref().unwrap();
        assert!(!hk.is_contiguous() && !hv.is_contiguous(), "kept as a view");
        let beside = narrow_snapshot(&snap(Vec::new()), 4).unwrap();
        assert!(beside.layers[0].as_ref().unwrap().0.is_contiguous());

        let pc = PrefixCache::new(true, 8, 0, 4, 8192, 0);
        let kv_store = KvCacheStore::new(std::time::Duration::from_secs(600));
        assert_eq!(
            pc.hydrate_request_from_snapshot(&kv_store, "m", "req", &host, &[])
                .unwrap(),
            4
        );
        let entry = kv_store
            .get_entry(&KvCacheStore::cache_key("m", "req"))
            .unwrap();
        let kv = entry.layers[0].as_ref().unwrap();
        let flat = |t: Tensor| t.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let want = |t: &Tensor| flat(t.narrow(2, 0, 4).unwrap().contiguous().unwrap());
        assert_eq!(flat(kv.k().unwrap().unwrap()), want(&k));
        assert_eq!(flat(kv.v().unwrap().unwrap()), want(&v));
    }

    /// #257: only a card reached through WSL2 moves its snapshots off the
    /// card by default; the switch decides for any card, never the processor.
    #[test]
    fn a_snapshot_leaves_the_card_by_default_only_under_wsl2() {
        use SnapshotHome::{BesideLive, HostRam};
        assert_eq!(SnapshotHome::choose(true, true, None), HostRam);
        assert_eq!(SnapshotHome::choose(true, false, None), BesideLive);
        assert_eq!(
            SnapshotHome::choose(true, true, Some(BesideLive)),
            BesideLive
        );
        assert_eq!(SnapshotHome::choose(true, false, Some(HostRam)), HostRam);
        assert_eq!(SnapshotHome::choose(false, true, Some(HostRam)), BesideLive);
        assert_eq!(SnapshotHome::choose(false, true, None), BesideLive);
    }

    /// A snapshot OWNS its positions. Narrowed along its positions, a cache
    /// with ONE key/value head is still contiguous to candle, whose
    /// `contiguous()` then hands back the live buffer itself — so the next
    /// in-place write (a window's slide, a rewind and rewrite) changed the
    /// saved prompt under the next turn that read it.
    #[test]
    fn a_snapshot_of_a_one_head_cache_does_not_share_the_live_buffer() {
        let dev = Device::Cpu;
        let k = Tensor::arange(0f32, 32.0, &dev)
            .unwrap()
            .reshape((1usize, 1, 8, 4))
            .unwrap();
        let v = (&k + 1000.0).unwrap();
        let mut kv = LayerKv::with_capacity(2, 64, 64);
        kv.append(&k, &v).unwrap();
        let mut layers = vec![Some(kv)];
        let snap = snapshot_at(&layers, &[], 8, 2, 4096, SnapshotHome::BesideLive).unwrap();

        // The live cache rewinds and writes other positions where the saved
        // ones were.
        let live = layers[0].as_mut().unwrap();
        live.truncate(0);
        let other = Tensor::zeros((1usize, 1, 8, 4), candle_core::DType::F32, &dev).unwrap();
        live.append(&other, &other).unwrap();

        let flat = |t: &Tensor| t.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let (sk, sv) = snap.layers[0].as_ref().unwrap();
        assert_eq!(flat(sk), flat(&k));
        assert_eq!(flat(sv), flat(&v));
    }

    /// On the processor the live cache already is host RAM: a snapshot asked
    /// to go there is never marked as moved, so it stays charged to the
    /// processor's KV budget (gotcha #440) rather than escaping it.
    #[test]
    fn the_processors_snapshot_stays_charged_beside_its_live_cache() {
        assert_eq!(
            SnapshotHome::for_device(&Device::Cpu),
            SnapshotHome::BesideLive
        );
        let kv_store = KvCacheStore::new(std::time::Duration::from_secs(600));
        make_fake_kv(&kv_store, "m", "req-a", 2, 10);
        let key = KvCacheStore::cache_key("m", "req-a");
        let entry = kv_store.get_entry(&key).unwrap();
        let snap = snapshot_at(&entry.layers, &[], 10, 2, 4096, SnapshotHome::HostRam).unwrap();
        assert!(snap.restore_to.iter().all(Option::is_none));
        assert_eq!(snap.home(), SnapshotHome::BesideLive);
        assert_eq!(snap.bytes_in(SnapshotHome::BesideLive), snap.bytes());
    }

    #[test]
    fn lookup_miss_when_disabled() {
        let pc = PrefixCache::new(false, 8, 32, 8, 8192, 0);
        assert!(pc.lookup("m", &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).is_none());
    }

    #[test]
    fn lookup_miss_below_min_tokens() {
        let pc = PrefixCache::new(true, 8, 32, 8, 8192, 0);
        assert!(pc.lookup("m", &[1, 2, 3]).is_none());
    }

    #[test]
    fn insert_and_lookup_exact_prefix() {
        let pc = PrefixCache::new(true, 8, 0, 4, 8192, 0);
        let kv_store = KvCacheStore::new(std::time::Duration::from_secs(600));
        make_fake_kv(&kv_store, "m", "req-a", 2, 10);

        let tokens: Vec<u32> = (1..=10).collect();
        pc.insert_from_kv(
            "m",
            "req-a",
            &kv_store,
            &tokens,
            ANYWHERE,
            KeepAsked::Unasked,
        );
        assert_eq!(pc.entry_count("m"), 1);

        // New request with same tokens + 5 more: should hit at 10.
        let new_tokens: Vec<u32> = (1..=15).collect();
        let snap = pc.lookup("m", &new_tokens).expect("hit");
        assert_eq!(snap.token_count, 10);
    }

    /// FUTURE_WORK #10: a split segment's prompt, stored under the
    /// coordinator's block chain, is restored by a later turn whose chain
    /// begins with it — exactly the positions asked for — and misses on more
    /// than was stored, a part of a block, another block size or another
    /// opening. What is kept is whole blocks of the room it is given.
    #[test]
    fn a_split_prompt_is_restored_by_its_block_chain() {
        let pc = PrefixCache::new(true, 8, 0, 4, 0, 0);
        let kv_store = KvCacheStore::new(std::time::Duration::from_secs(600));
        make_fake_kv(&kv_store, "seg", "turn-1", 2, 40);
        // Six keys of 8 positions; the cache holds 40 positions: five blocks.
        let keys: Vec<[u8; 32]> = (0..6u8).map(|i| [i; 32]).collect();
        assert_eq!(
            pc.insert_chain_from_kv("seg", "turn-1", &kv_store, 8, &keys, ANYWHERE),
            5
        );
        assert_eq!(
            pc.lookup_chain("seg", 8, &keys, 24)
                .expect("hit")
                .token_count,
            24
        );
        assert_eq!(
            pc.lookup_chain("seg", 8, &keys, 40)
                .expect("all of it")
                .token_count,
            40
        );
        assert!(
            pc.lookup_chain("seg", 8, &keys, 48).is_none(),
            "more than stored"
        );
        assert!(
            pc.lookup_chain("seg", 8, &keys, 20).is_none(),
            "not whole blocks"
        );
        assert!(
            pc.lookup_chain("seg", 16, &keys, 32).is_none(),
            "another block size"
        );
        let mut other = keys.clone();
        other[1] = [99; 32];
        assert!(
            pc.lookup_chain("seg", 8, &other, 16).is_none(),
            "another opening"
        );
        assert!(
            pc.lookup_chain("seg", 8, &other, 8).is_some(),
            "the shared first block"
        );
        // The room beside the live cache bounds what is kept, in whole blocks.
        make_fake_kv(&kv_store, "seg", "turn-2", 2, 40);
        assert_eq!(
            pc.insert_chain_from_kv("seg", "turn-2", &kv_store, 8, &other, beside(20)),
            2
        );
        // A request with no cache keeps nothing.
        assert_eq!(
            pc.insert_chain_from_kv("seg", "none", &kv_store, 8, &keys, ANYWHERE),
            0
        );
    }

    /// Chain entries and token entries share one cache without disturbing
    /// each other: a token prompt never matches a chain entry, a token insert
    /// never drops one (every prompt "starts with" its empty token list), and
    /// a growing split conversation keeps ONE chain entry.
    #[test]
    fn chain_entries_and_token_entries_share_the_cache() {
        let pc = PrefixCache::new(true, 8, 0, 4, 0, 0);
        let kv_store = KvCacheStore::new(std::time::Duration::from_secs(600));
        make_fake_kv(&kv_store, "m", "seg", 2, 16);
        let keys = vec![[1u8; 32], [2u8; 32]];
        assert_eq!(
            pc.insert_chain_from_kv("m", "seg", &kv_store, 8, &keys, ANYWHERE),
            2
        );
        make_fake_kv(&kv_store, "m", "tok", 2, 10);
        let tokens: Vec<u32> = (1..=10).collect();
        pc.insert_from_kv("m", "tok", &kv_store, &tokens, ANYWHERE, KeepAsked::Unasked);
        assert_eq!(
            pc.entry_count("m"),
            2,
            "the token insert kept the chain entry"
        );
        assert!(pc.lookup_chain("m", 8, &keys, 16).is_some());
        let longer_prompt: Vec<u32> = (1..=15).collect();
        assert_eq!(
            pc.lookup("m", &longer_prompt)
                .expect("token hit")
                .token_count,
            10
        );
        make_fake_kv(&kv_store, "m", "seg-2", 2, 24);
        let grown = vec![[1u8; 32], [2u8; 32], [3u8; 32]];
        assert_eq!(
            pc.insert_chain_from_kv("m", "seg-2", &kv_store, 8, &grown, ANYWHERE),
            3
        );
        assert_eq!(
            pc.entry_count("m"),
            2,
            "the grown chain replaced the one it extends"
        );
        assert_eq!(
            pc.lookup_chain("m", 8, &grown, 24)
                .expect("hit")
                .token_count,
            24
        );
    }

    #[test]
    fn partial_match_narrows_to_the_full_shared_prefix() {
        // One entry at the full prompt length; a partially-overlapping prompt
        // hits at the WHOLE shared prefix (6), not a block boundary (the old
        // per-boundary storage could only offer 4).
        let pc = PrefixCache::new(true, 16, 4, 4, 8192, 0);
        let kv_store = KvCacheStore::new(std::time::Duration::from_secs(600));
        make_fake_kv(&kv_store, "m", "req-a", 2, 10);

        let tokens_a: Vec<u32> = (1..=10).collect();
        pc.insert_from_kv(
            "m",
            "req-a",
            &kv_store,
            &tokens_a,
            ANYWHERE,
            KeepAsked::Unasked,
        );
        assert_eq!(pc.entry_count("m"), 1);

        let mut tokens_b: Vec<u32> = (1..=6).collect();
        tokens_b.extend_from_slice(&[99, 99, 99, 99]);
        let snap = pc.lookup("m", &tokens_b).expect("hit");
        assert_eq!(snap.token_count, 6);
        // The narrowed snapshot's tensors must actually be 6 positions long —
        // hydration appends them verbatim.
        for layer in snap.layers.iter().flatten() {
            assert_eq!(layer.0.dims()[2], 6);
            assert_eq!(layer.1.dims()[2], 6);
        }
    }

    #[test]
    fn one_snapshot_per_insert_even_with_many_block_boundaries() {
        // Gotcha #312: this used to create one entry PER block boundary
        // (3 here; 45 for a real 2.9k-token prompt), each an independent
        // full-prefix KV copy — quadratic time and memory.
        let pc = PrefixCache::new(true, 16, 4, 4, 8192, 0);
        let kv_store = KvCacheStore::new(std::time::Duration::from_secs(600));
        make_fake_kv(&kv_store, "m", "req-a", 2, 12);
        let tokens: Vec<u32> = (1..=12).collect();
        let manifest = pc.insert_from_kv(
            "m",
            "req-a",
            &kv_store,
            &tokens,
            ANYWHERE,
            KeepAsked::Unasked,
        );
        assert_eq!(pc.entry_count("m"), 1);
        // The announce manifest still covers every block boundary.
        assert_eq!(manifest.len(), 3);
    }

    #[test]
    fn repeated_identical_insert_skips_the_copy() {
        let pc = PrefixCache::new(true, 16, 4, 4, 8192, 0);
        let kv_store = KvCacheStore::new(std::time::Duration::from_secs(600));
        let tokens: Vec<u32> = (1..=8).collect();

        // req-a's KV is zeros (make_fake_kv); insert it.
        make_fake_kv(&kv_store, "m", "req-a", 2, 8);
        let m1 = pc.insert_from_kv(
            "m",
            "req-a",
            &kv_store,
            &tokens,
            ANYWHERE,
            KeepAsked::Unasked,
        );

        // req-b holds DIFFERENT KV content (ones) for the same tokens. If the
        // second insert re-snapshots, the cached tensors become ones; if it
        // correctly skips (the prefix is already covered), they stay zeros.
        {
            let device = Device::Cpu;
            let mut entry = kv_store.get_or_create("m", "req-b", 2);
            for slot in entry.layers.iter_mut() {
                let k = Tensor::ones((1usize, 1, 8, 4), DType::F32, &device).unwrap();
                let v = Tensor::ones((1usize, 1, 8, 4), DType::F32, &device).unwrap();
                let mut kv = LayerKv::with_capacity(2, 4096, 4096);
                kv.append(&k, &v).unwrap();
                *slot = Some(kv);
            }
        }
        let m2 = pc.insert_from_kv(
            "m",
            "req-b",
            &kv_store,
            &tokens,
            ANYWHERE,
            KeepAsked::Unasked,
        );

        assert_eq!(pc.entry_count("m"), 1);
        // The skip must still report the full manifest, or the caller stops
        // announcing blocks it can serve.
        assert_eq!(m1.len(), m2.len());
        let lookup_tokens: Vec<u32> = (1..=10).collect();
        let snap = pc.lookup("m", &lookup_tokens).expect("hit");
        let (k, _) = snap.layers[0].as_ref().expect("layer");
        let vals: Vec<f32> = k.flatten_all().unwrap().to_vec1().unwrap();
        assert!(
            vals.iter().all(|v| *v == 0.0),
            "second insert replaced the snapshot instead of skipping"
        );
    }

    #[test]
    fn growing_conversation_keeps_one_entry() {
        // Turn 2's prompt extends turn 1's, so turn 1's entry is fully
        // covered by the new one and must be pruned, not accumulated.
        let pc = PrefixCache::new(true, 16, 4, 4, 8192, 0);
        let kv_store = KvCacheStore::new(std::time::Duration::from_secs(600));
        make_fake_kv(&kv_store, "m", "req-a", 2, 8);
        pc.insert_from_kv(
            "m",
            "req-a",
            &kv_store,
            &(1..=8).collect::<Vec<_>>(),
            ANYWHERE,
            KeepAsked::Unasked,
        );
        make_fake_kv(&kv_store, "m", "req-b", 2, 12);
        pc.insert_from_kv(
            "m",
            "req-b",
            &kv_store,
            &(1..=12).collect::<Vec<_>>(),
            ANYWHERE,
            KeepAsked::Unasked,
        );

        assert_eq!(pc.entry_count("m"), 1);
        let snap = pc.lookup("m", &(1..=14).collect::<Vec<_>>()).expect("hit");
        assert_eq!(snap.token_count, 12);
    }

    /// A request's cache whose second layer holds only half the positions of
    /// its first: `snapshot_at` refuses it, so an insert of it fails at the
    /// copy — after anything the insert does before the copy.
    fn make_kv_that_fails_the_copy(kv_store: &KvCacheStore, model_key: &str, request_id: &str) {
        let device = Device::Cpu;
        let mut entry = kv_store.get_or_create(model_key, request_id, 2);
        for (slot, seq_len) in entry.layers.iter_mut().zip([16usize, 8]) {
            let k = Tensor::zeros((1usize, 1, seq_len, 4), DType::F32, &device).unwrap();
            let v = Tensor::zeros((1usize, 1, seq_len, 4), DType::F32, &device).unwrap();
            let mut kv = LayerKv::with_capacity(2, 4096, 4096);
            kv.append(&k, &v).unwrap();
            *slot = Some(kv);
        }
    }

    /// #367: the snapshot a growing conversation's next one replaces is let go
    /// BEFORE the copy, so the save never holds the prompt twice and the copy
    /// reuses its memory. Seen from outside through a copy that fails: the
    /// entry it would have replaced is already gone.
    #[test]
    fn a_replaced_snapshot_is_let_go_before_the_copy() {
        let pc = PrefixCache::new(true, 16, 4, 4, 8192, 0);
        let kv_store = KvCacheStore::new(std::time::Duration::from_secs(600));
        make_fake_kv(&kv_store, "m", "turn-1", 2, 8);
        let turn_1: Vec<u32> = (1..=8).collect();
        pc.insert_from_kv(
            "m",
            "turn-1",
            &kv_store,
            &turn_1,
            ANYWHERE,
            KeepAsked::Unasked,
        );
        // Another conversation's entry, which turn 2 does not replace.
        make_fake_kv(&kv_store, "m", "other", 2, 8);
        let other: Vec<u32> = (101..=108).collect();
        pc.insert_from_kv(
            "m",
            "other",
            &kv_store,
            &other,
            ANYWHERE,
            KeepAsked::Unasked,
        );
        assert_eq!(pc.entry_count("m"), 2);

        make_kv_that_fails_the_copy(&kv_store, "m", "turn-2");
        let turn_2: Vec<u32> = (1..=16).collect();
        let manifest = pc.insert_from_kv(
            "m",
            "turn-2",
            &kv_store,
            &turn_2,
            ANYWHERE,
            KeepAsked::Unasked,
        );
        assert!(manifest.is_empty(), "the copy failed");
        assert_eq!(
            pc.entry_count("m"),
            1,
            "turn 1's snapshot went before the copy"
        );
        assert!(pc.lookup("m", &turn_1).is_none());
        assert!(pc.lookup("m", &[other.as_slice(), &[1]].concat()).is_some());
    }

    /// #367 (c): the room is planned once the save knows what it replaces —
    /// the planner is handed the previous turn's bytes, in the memory it is
    /// kept in, and never another conversation's — and the plan's own
    /// eviction is made only by the insert, beyond that. Counted as held, the
    /// previous turn made a tight budget evict the other conversation for
    /// room it was about to give back.
    #[test]
    fn the_room_is_planned_with_what_the_save_replaces() {
        let pc = PrefixCache::new(true, 16, 4, 4, 8192, 0);
        let kv_store = KvCacheStore::new(std::time::Duration::from_secs(600));
        make_fake_kv(&kv_store, "m", "turn-1", 2, 8);
        let turn_1: Vec<u32> = (1..=8).collect();
        pc.insert_from_kv(
            "m",
            "turn-1",
            &kv_store,
            &turn_1,
            ANYWHERE,
            KeepAsked::Unasked,
        );
        let turn_1_bytes = pc.bytes_in(SnapshotHome::BesideLive) as u64;
        // Another conversation, older than turn 1's next.
        make_fake_kv(&kv_store, "m", "other", 2, 8);
        let other: Vec<u32> = (101..=108).collect();
        pc.insert_from_kv(
            "m",
            "other",
            &kv_store,
            &other,
            ANYWHERE,
            KeepAsked::Unasked,
        );

        make_fake_kv(&kv_store, "m", "turn-2", 2, 16);
        let turn_2: Vec<u32> = (1..=16).collect();
        let handed = std::cell::Cell::new(None);
        let manifest = pc.insert_from_kv(
            "m",
            "turn-2",
            &kv_store,
            &turn_2,
            |let_go: LetGo| {
                handed.set(Some(let_go));
                ANYWHERE
            },
            KeepAsked::Unasked,
        );
        assert!(!manifest.is_empty());
        assert_eq!(
            handed.get(),
            Some(LetGo {
                beside_live: turn_1_bytes,
                host_ram: 0,
            }),
            "the plan is told the previous turn goes, and nothing else"
        );
        assert_eq!(pc.entry_count("m"), 2, "turn 1 replaced, the other kept");
        assert!(pc.lookup("m", &[other.as_slice(), &[1]].concat()).is_some());
    }

    /// #367 (c): a prompt already saved copies nothing, so it asks for no
    /// room and gives up nothing — the plan, which evicted before the insert
    /// looked, took other conversations' prompts for a copy then skipped.
    #[test]
    fn a_prompt_already_saved_asks_for_no_room() {
        let pc = PrefixCache::new(true, 16, 4, 4, 8192, 0);
        let kv_store = KvCacheStore::new(std::time::Duration::from_secs(600));
        for (req, base) in [("other", 100u32), ("again", 200)] {
            make_fake_kv(&kv_store, "m", req, 2, 8);
            let tokens: Vec<u32> = (base..base + 8).collect();
            pc.insert_from_kv("m", req, &kv_store, &tokens, ANYWHERE, KeepAsked::Unasked);
        }
        make_fake_kv(&kv_store, "m", "again-2", 2, 8);
        let again: Vec<u32> = (200..208).collect();
        let manifest = pc.insert_from_kv(
            "m",
            "again-2",
            &kv_store,
            &again,
            |_: LetGo| -> SnapshotRoom { panic!("a covered prompt planned room") },
            KeepAsked::Unasked,
        );
        assert!(!manifest.is_empty(), "the covering entry's blocks");
        assert_eq!(pc.entry_count("m"), 2, "nothing given up");

        // The same on a split's stored prompt (#10).
        make_fake_kv(&kv_store, "seg", "turn-1", 2, 16);
        let keys = vec![[3u8; 32]; 2];
        assert_eq!(
            pc.insert_chain_from_kv("seg", "turn-1", &kv_store, 8, &keys, ANYWHERE),
            2
        );
        make_fake_kv(&kv_store, "seg", "turn-1b", 2, 16);
        assert_eq!(
            pc.insert_chain_from_kv(
                "seg",
                "turn-1b",
                &kv_store,
                8,
                &keys,
                |_: LetGo| -> SnapshotRoom { panic!("a covered chain planned room") }
            ),
            2
        );
    }

    /// #367 (c): the plan's eviction is made in the memory the copy goes to,
    /// least recently hit first across every model, counting what the caps
    /// already gave back there — never in the other memory.
    #[test]
    fn a_planned_eviction_is_made_in_the_home_the_copy_goes_to() {
        let pc = PrefixCache::new(true, 16, 4, 4, 8192, 0);
        let kv_store = KvCacheStore::new(std::time::Duration::from_secs(600));
        // Oldest first: one off the card (another model), then two beside it.
        for (model, req, base) in [
            ("n", "host", 100u32),
            ("n", "card-old", 200),
            ("m", "card-new", 300),
        ] {
            make_fake_kv(&kv_store, model, req, 2, 8);
            let tokens: Vec<u32> = (base..base + 8).collect();
            pc.insert_from_kv(model, req, &kv_store, &tokens, ANYWHERE, KeepAsked::Unasked);
        }
        move_off_card(&pc, "n", 100);
        let per_entry = pc.bytes_in(SnapshotHome::HostRam) as u64;
        assert!(per_entry > 0);

        make_fake_kv(&kv_store, "m", "new", 2, 8);
        let new: Vec<u32> = (400..408).collect();
        let room = SnapshotRoom {
            home: SnapshotHome::BesideLive,
            max_positions: usize::MAX,
            evict_bytes: per_entry,
        };
        assert!(!pc
            .insert_from_kv("m", "new", &kv_store, &new, room, KeepAsked::Unasked)
            .is_empty());
        assert!(
            pc.lookup("n", &(200..209).collect::<Vec<u32>>()).is_none(),
            "the oldest copy beside the card went, though it is another model's"
        );
        assert!(pc.lookup("m", &(300..309).collect::<Vec<u32>>()).is_some());
        assert_eq!(
            pc.bytes_in(SnapshotHome::HostRam) as u64,
            per_entry,
            "the older copy in host memory gave the card nothing"
        );
    }

    /// And the caps: a full bucket's least recently hit entry goes BEFORE the
    /// copy too (llama.cpp's "make room before allocating"), not after it — the
    /// entry the insert's own pass would have dropped once the copy was in.
    #[test]
    fn a_full_cache_evicts_before_the_copy() {
        let pc = PrefixCache::new(true, 2, 4, 4, 8192, 0);
        let kv_store = KvCacheStore::new(std::time::Duration::from_secs(600));
        for (req, base) in [("a", 100u32), ("b", 200)] {
            make_fake_kv(&kv_store, "m", req, 2, 8);
            let tokens: Vec<u32> = (base..base + 8).collect();
            pc.insert_from_kv("m", req, &kv_store, &tokens, ANYWHERE, KeepAsked::Unasked);
        }
        // "b" is hit, so "a" is the least recently hit.
        assert!(pc.lookup("m", &(200..209).collect::<Vec<u32>>()).is_some());
        make_kv_that_fails_the_copy(&kv_store, "m", "c");
        let c: Vec<u32> = (300..316).collect();
        assert!(pc
            .insert_from_kv("m", "c", &kv_store, &c, ANYWHERE, KeepAsked::Unasked)
            .is_empty());
        assert_eq!(pc.entry_count("m"), 1, "the oldest went before the copy");
        assert!(pc.lookup("m", &(100..109).collect::<Vec<u32>>()).is_none());
        assert!(pc.lookup("m", &(200..209).collect::<Vec<u32>>()).is_some());
    }

    /// The same on a split's stored prompt (#10): the chain entry ours extends
    /// goes before the copy; another block size's stays.
    #[test]
    fn a_replaced_split_prompt_is_let_go_before_the_copy() {
        let pc = PrefixCache::new(true, 8, 0, 4, 0, 0);
        let kv_store = KvCacheStore::new(std::time::Duration::from_secs(600));
        let keys: Vec<[u8; 32]> = (0..4u8).map(|i| [i; 32]).collect();
        make_fake_kv(&kv_store, "seg", "turn-1", 2, 8);
        assert_eq!(
            pc.insert_chain_from_kv("seg", "turn-1", &kv_store, 8, &keys, ANYWHERE),
            1
        );
        assert_eq!(
            pc.insert_chain_from_kv("seg", "turn-1", &kv_store, 4, &keys, ANYWHERE),
            2
        );
        make_kv_that_fails_the_copy(&kv_store, "seg", "turn-2");
        assert_eq!(
            pc.insert_chain_from_kv("seg", "turn-2", &kv_store, 8, &keys, ANYWHERE),
            0,
            "the copy failed"
        );
        assert!(
            pc.lookup_chain("seg", 8, &keys, 8).is_none(),
            "gone before the copy"
        );
        assert!(pc.lookup_chain("seg", 4, &keys, 8).is_some());
    }

    #[test]
    fn identical_prompt_hits_at_len_minus_one() {
        // Re-asking the exact cached prompt used to MISS (the entry could
        // not leave a token to forward). Narrowing serves it at len - 1.
        let pc = PrefixCache::new(true, 16, 4, 4, 8192, 0);
        let kv_store = KvCacheStore::new(std::time::Duration::from_secs(600));
        make_fake_kv(&kv_store, "m", "req-a", 2, 10);
        let tokens: Vec<u32> = (1..=10).collect();
        pc.insert_from_kv(
            "m",
            "req-a",
            &kv_store,
            &tokens,
            ANYWHERE,
            KeepAsked::Unasked,
        );

        let snap = pc.lookup("m", &tokens).expect("hit");
        assert_eq!(snap.token_count, 9);
    }

    #[test]
    fn miss_when_no_shared_prefix() {
        let pc = PrefixCache::new(true, 8, 0, 4, 8192, 0);
        let kv_store = KvCacheStore::new(std::time::Duration::from_secs(600));
        make_fake_kv(&kv_store, "m", "req-a", 2, 10);
        let tokens_a: Vec<u32> = (1..=10).collect();
        pc.insert_from_kv(
            "m",
            "req-a",
            &kv_store,
            &tokens_a,
            ANYWHERE,
            KeepAsked::Unasked,
        );

        let tokens_b: Vec<u32> = vec![99, 98, 97, 96, 95, 94, 93, 92];
        assert!(pc.lookup("m", &tokens_b).is_none());
    }

    #[test]
    fn hydrate_copies_snapshot_into_kv_store() {
        let pc = PrefixCache::new(true, 8, 0, 4, 8192, 0);
        let kv_store = KvCacheStore::new(std::time::Duration::from_secs(600));
        make_fake_kv(&kv_store, "m", "req-a", 2, 10);
        let tokens: Vec<u32> = (1..=10).collect();
        pc.insert_from_kv(
            "m",
            "req-a",
            &kv_store,
            &tokens,
            ANYWHERE,
            KeepAsked::Unasked,
        );
        // lookup requires at least one token left to forward, so query with a
        // longer prompt that still has `tokens` as its strict prefix.
        let lookup_tokens: Vec<u32> = (1..=12).collect();
        let snap = pc.lookup("m", &lookup_tokens).expect("hit");

        let seeded = pc
            .hydrate_request_from_snapshot(&kv_store, "m", "req-b", &snap, &[])
            .unwrap();
        assert_eq!(seeded, 10);

        // req-b now has a KV entry with 10 positions per layer.
        let key = KvCacheStore::cache_key("m", "req-b");
        let entry = kv_store.get_entry(&key).unwrap();
        for l in entry.layers.iter() {
            let kv = l.as_ref().expect("layer populated");
            assert_eq!(kv.current_seq_len(), 10);
        }
    }

    #[test]
    fn block_hash_chain_matches_for_shared_prefix() {
        // Two prompts that share the first two 4-token blocks must have
        // identical block_hash[0..2]. The third block diverges so
        // block_hash[2] must differ.
        let a: Vec<u32> = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
        let b: Vec<u32> = vec![1, 2, 3, 4, 5, 6, 7, 8, 99, 99, 99, 99];
        let ha = compute_block_hashes(&a, 4);
        let hb = compute_block_hashes(&b, 4);
        assert_eq!(ha.len(), 3);
        assert_eq!(hb.len(), 3);
        assert_eq!(ha[0].block_hash, hb[0].block_hash);
        assert_eq!(ha[1].block_hash, hb[1].block_hash);
        assert_ne!(ha[2].block_hash, hb[2].block_hash);
        assert_eq!(ha[0].token_count, 4);
        assert_eq!(ha[1].token_count, 8);
        assert_eq!(ha[2].token_count, 12);
    }

    #[test]
    fn block_hash_chain_handles_partial_trailing_block() {
        // Last 3 tokens are incomplete and must NOT be hashed (block_size=4).
        let toks: Vec<u32> = (0..11).collect();
        let h = compute_block_hashes(&toks, 4);
        assert_eq!(h.len(), 2);
        assert_eq!(h[1].token_count, 8);
    }

    #[test]
    fn block_hash_chain_zero_block_size_is_empty() {
        let h = compute_block_hashes(&[1, 2, 3, 4], 0);
        assert!(h.is_empty());
    }

    #[test]
    fn insert_from_kv_returns_block_manifest() {
        // block=4 → 12 tokens → 3 blocks
        let pc = PrefixCache::new(true, 8, 4, 4, 8192, 0);
        let kv_store = KvCacheStore::new(std::time::Duration::from_secs(600));
        make_fake_kv(&kv_store, "m", "req-a", 2, 12);
        let tokens: Vec<u32> = (1..=12).collect();
        let manifest = pc.insert_from_kv(
            "m",
            "req-a",
            &kv_store,
            &tokens,
            ANYWHERE,
            KeepAsked::Unasked,
        );
        assert_eq!(manifest.len(), 3);
        // token_count grows by block_size (4) each entry.
        assert_eq!(manifest[0].token_count, 4);
        assert_eq!(manifest[1].token_count, 8);
        assert_eq!(manifest[2].token_count, 12);
    }

    #[test]
    fn enumerate_manifest_dedups_across_entries() {
        // Two entries whose tokens share a 4-token prefix. They each insert
        // their own snapshot, but the chained hash for the shared block is
        // identical → manifest must dedup.
        let pc = PrefixCache::new(true, 8, 4, 4, 8192, 0);
        let kv_store = KvCacheStore::new(std::time::Duration::from_secs(600));
        make_fake_kv(&kv_store, "m", "req-a", 2, 8);
        make_fake_kv(&kv_store, "m", "req-b", 2, 8);
        let a: Vec<u32> = vec![1, 2, 3, 4, 50, 51, 52, 53];
        let b: Vec<u32> = vec![1, 2, 3, 4, 60, 61, 62, 63];
        let _ = pc.insert_from_kv("m", "req-a", &kv_store, &a, ANYWHERE, KeepAsked::Unasked);
        let _ = pc.insert_from_kv("m", "req-b", &kv_store, &b, ANYWHERE, KeepAsked::Unasked);
        let manifest = pc.enumerate_manifest("m");
        // Block 0 (shared) appears once, plus block 1 from each prompt = 3.
        assert_eq!(manifest.len(), 3);
        // Compare against direct chain hashes to confirm dedup logic.
        let ha = compute_block_hashes(&a, 4);
        let hb = compute_block_hashes(&b, 4);
        assert!(manifest.iter().any(|e| e.block_hash == ha[0].block_hash));
        assert!(manifest.iter().any(|e| e.block_hash == ha[1].block_hash));
        assert!(manifest.iter().any(|e| e.block_hash == hb[1].block_hash));
    }

    #[test]
    fn enumerate_manifest_zero_block_size_is_empty() {
        let pc = PrefixCache::new(true, 8, 0, 4, 8192, 0);
        let kv_store = KvCacheStore::new(std::time::Duration::from_secs(600));
        make_fake_kv(&kv_store, "m", "req-a", 2, 8);
        let _ = pc.insert_from_kv(
            "m",
            "req-a",
            &kv_store,
            &(1..=8).collect::<Vec<_>>(),
            ANYWHERE,
            KeepAsked::Unasked,
        );
        // block_tokens=0 → enumerate_manifest is empty even though the
        // entry exists (it was inserted via the always-store-tail rule).
        assert!(pc.enumerate_manifest("m").is_empty());
    }

    #[test]
    fn verify_token_hash_chain_matches_manifest() {
        let tokens: Vec<u32> = (1..=12).collect();
        let manifest = compute_block_hashes(&tokens, 4);
        // Each manifest entry should verify against its own final block.
        for entry in &manifest {
            assert!(verify_token_hash_chain(
                &tokens,
                4,
                entry.token_count as usize,
                &entry.block_hash,
            ));
        }
    }

    #[test]
    fn verify_token_hash_chain_rejects_wrong_tokens() {
        let tokens_a: Vec<u32> = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let manifest = compute_block_hashes(&tokens_a, 4);
        let last = manifest.last().unwrap();
        // Different tokens must NOT verify against the original hash.
        let tokens_b: Vec<u32> = vec![9, 9, 9, 9, 9, 9, 9, 9];
        assert!(!verify_token_hash_chain(
            &tokens_b,
            4,
            last.token_count as usize,
            &last.block_hash,
        ));
    }

    #[test]
    fn verify_token_hash_chain_rejects_non_boundary() {
        let tokens: Vec<u32> = (1..=8).collect();
        let h = compute_block_hashes(&tokens, 4);
        let last = h.last().unwrap();
        // Off-boundary count (e.g. 5 with block_size=4) must be rejected.
        assert!(!verify_token_hash_chain(&tokens, 4, 5, &last.block_hash));
    }

    #[test]
    fn snapshot_roundtrip_preserves_tensors() {
        use candle_core::{DType, Device};
        let device = Device::Cpu;
        // Construct a tiny snapshot by hand — 2 layers, seq=4, head_dim=3.
        let k0 = Tensor::from_vec(
            (0..24).map(|i| i as f32).collect::<Vec<_>>(),
            (1usize, 2, 4, 3),
            &device,
        )
        .unwrap();
        let v0 = Tensor::from_vec(
            (0..24).map(|i| (i * 2) as f32).collect::<Vec<_>>(),
            (1usize, 2, 4, 3),
            &device,
        )
        .unwrap();
        let k1 = Tensor::from_vec(
            (0..24).map(|i| (i + 100) as f32).collect::<Vec<_>>(),
            (1usize, 2, 4, 3),
            &device,
        )
        .unwrap();
        let v1 = Tensor::from_vec(
            (0..24).map(|i| (i + 200) as f32).collect::<Vec<_>>(),
            (1usize, 2, 4, 3),
            &device,
        )
        .unwrap();
        let snap = KvSnapshot {
            token_count: 4,
            layers: vec![Some((k0, v0)), Some((k1, v1))],
            dim: 2,
            max_seq_len: 4096,
            restore_to: Vec::new(),
            held_from: Vec::new(),
            serves_from: 0,
            ssm: Vec::new(),
        };
        let tokens: Vec<u32> = vec![7, 8, 9, 10];
        let bytes = serialize_snapshot_with_block_size(&snap, &tokens, None).expect("serialize");
        let (decoded, decoded_tokens) = deserialize_snapshot(&bytes, &device).expect("deserialize");
        assert_eq!(decoded.token_count, 4);
        assert_eq!(decoded.dim, 2);
        assert_eq!(decoded.max_seq_len, 4096);
        assert_eq!(decoded_tokens, tokens);
        assert_eq!(decoded.layers.len(), 2);
        for (orig, got) in snap.layers.iter().zip(decoded.layers.iter()) {
            let (ok, ov) = orig.as_ref().expect("orig some");
            let (dk, dv) = got.as_ref().expect("decoded some");
            assert_eq!(ok.shape().dims(), dk.shape().dims());
            assert_eq!(ov.shape().dims(), dv.shape().dims());
            let ok_vec: Vec<f32> = ok.flatten_all().unwrap().to_vec1().unwrap();
            let dk_vec: Vec<f32> = dk
                .to_dtype(DType::F32)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1()
                .unwrap();
            assert_eq!(ok_vec, dk_vec);
            let ov_vec: Vec<f32> = ov.flatten_all().unwrap().to_vec1().unwrap();
            let dv_vec: Vec<f32> = dv
                .to_dtype(DType::F32)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1()
                .unwrap();
            assert_eq!(ov_vec, dv_vec);
        }
    }

    #[test]
    fn snapshot_rejects_bad_magic() {
        use candle_core::Device;
        let bad = vec![0u8; 32];
        assert!(deserialize_snapshot(&bad, &Device::Cpu).is_err());
    }

    #[test]
    fn snapshot_rejects_bad_version() {
        use candle_core::Device;
        let mut buf = Vec::new();
        buf.extend_from_slice(KV_SNAPSHOT_MAGIC);
        buf.extend_from_slice(&999u32.to_le_bytes());
        buf.extend_from_slice(&0u64.to_le_bytes());
        assert!(deserialize_snapshot(&buf, &Device::Cpu).is_err());
    }

    /// The sizes in a snapshot are the SENDER's. Wrapped, they passed the
    /// length checks and the slice panicked the task verifying a peer's
    /// snapshot (sweep 2026-10-24); each is refused instead.
    #[test]
    fn a_snapshot_whose_sizes_overflow_is_refused_not_a_panic() {
        use candle_core::Device;
        let frame = |header_len: u64, header: &[u8], body: &[u8]| {
            let mut buf = Vec::new();
            buf.extend_from_slice(KV_SNAPSHOT_MAGIC);
            buf.extend_from_slice(&KV_SNAPSHOT_VERSION.to_le_bytes());
            buf.extend_from_slice(&header_len.to_le_bytes());
            buf.extend_from_slice(header);
            buf.extend_from_slice(body);
            buf
        };
        // `16 + header_len` wraps to 7.
        let wrapped = frame(u64::MAX - 8, b"{}", &[0u8; 16]);
        assert!(deserialize_snapshot(&wrapped, &Device::Cpu).is_err());
        // `k_count * 4` wraps to 0, and the shape's product past usize.
        for meta in [
            SnapshotLayerMeta {
                k_shape: vec![1],
                v_shape: vec![1],
                k_count: 1 << 62,
            },
            SnapshotLayerMeta {
                k_shape: vec![1],
                v_shape: vec![usize::MAX, 2],
                k_count: 1,
            },
            // K's shape multiplies past usize round to exactly the 0 values sent.
            SnapshotLayerMeta {
                k_shape: vec![1 << (usize::BITS / 2), 1 << (usize::BITS / 2)],
                v_shape: vec![1],
                k_count: 0,
            },
        ] {
            let header = serde_json::to_vec(&SnapshotHeader {
                token_count: 1,
                tokens: vec![1],
                dim: 1,
                max_seq_len: 8,
                block_size: None,
                layers: vec![Some(meta)],
            })
            .unwrap();
            let buf = frame(header.len() as u64, &header, &[0u8; 8]);
            assert!(deserialize_snapshot(&buf, &Device::Cpu).is_err());
        }
    }

    /// A request whose caller asked to keep it on this computer (#251 (2b))
    /// is reused here and never leaves: its entry is neither announced nor
    /// exported, though a later prompt on this node still hits it.
    #[test]
    fn a_kept_requests_cache_is_reused_here_and_never_leaves() {
        let pc = PrefixCache::new(true, 8, 4, 4, 8192, 0);
        let kv_store = KvCacheStore::new(std::time::Duration::from_secs(600));
        make_fake_kv(&kv_store, "m", "kept", 2, 12);
        let tokens: Vec<u32> = (1..=12).collect();
        let announced = pc.insert_from_kv(
            "m",
            "kept",
            &kv_store,
            &tokens,
            ANYWHERE,
            KeepAsked::OnThisComputer,
        );
        assert!(
            announced.is_empty(),
            "a kept request's blocks are announced"
        );
        assert!(pc.enumerate_manifest("m").is_empty());
        let hashes = compute_block_hashes(&tokens, 4);
        for block in &hashes {
            assert!(
                pc.export_snapshot_bytes("m", &block.block_hash).is_none(),
                "a kept request's cache is exported to a peer"
            );
        }
        assert!(pc.lookup("m", &tokens).is_some(), "nor is it reused here");

        // A longer prompt that replaces it keeps the mark: the strictest
        // instruction wins.
        let longer: Vec<u32> = (1..=16).collect();
        make_fake_kv(&kv_store, "m", "later", 2, 16);
        let announced = pc.insert_from_kv(
            "m",
            "later",
            &kv_store,
            &longer,
            ANYWHERE,
            KeepAsked::Unasked,
        );
        assert!(
            announced.is_empty(),
            "the entry that replaced it lost its keep"
        );

        // The control: an unasked request's cache is announced and exported.
        let open = PrefixCache::new(true, 8, 4, 4, 8192, 0);
        let announced = open.insert_from_kv(
            "m",
            "kept",
            &kv_store,
            &tokens,
            ANYWHERE,
            KeepAsked::Unasked,
        );
        assert_eq!(announced.len(), 3);
        assert!(open
            .export_snapshot_bytes("m", &hashes[1].block_hash)
            .is_some());
    }

    /// A Qwen 3.5-shaped cache: layer 0 attends (`positions` positions),
    /// layer 1 is recurrent, its state after `state_at` tokens.
    fn make_recurrent_kv(
        kv_store: &KvCacheStore,
        request_id: &str,
        positions: usize,
        state_at: usize,
    ) {
        let device = Device::Cpu;
        let mut entry = kv_store.get_or_create("m", request_id, 2);
        let k = Tensor::ones((1usize, 1, positions, 4), DType::F32, &device).unwrap();
        let mut kv = LayerKv::with_capacity(2, 4096, 4096);
        kv.append(&k, &k).unwrap();
        entry.layers[0] = Some(kv);
        let values = |n: usize, at: f32| (0..n).map(|i| at + i as f32).collect::<Vec<f32>>();
        let slot = SsmSlot {
            conv_state: Tensor::from_vec(values(6, 1.0), (1usize, 3, 2), &device).unwrap(),
            recurrent_state: Tensor::from_vec(values(8, 10.0), (1usize, 2, 2, 2), &device).unwrap(),
        };
        entry.ssm_states[1] = Some(SsmState::new(slot, state_at));
    }

    fn recurrent_values(state: &SsmState) -> (Vec<f32>, Vec<f32>) {
        let read = |t: &Tensor| t.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        (
            read(&state.current().conv_state),
            read(&state.current().recurrent_state),
        )
    }

    /// FUTURE_WORK #228: a recurrent layer's state summarises every token it
    /// read and cannot be cut back, so its snapshot serves a prompt that
    /// repeats ALL of it — never one sharing less, which an attention-only
    /// snapshot would serve narrowed — and it never leaves this node.
    #[test]
    fn a_recurrent_snapshot_is_restored_whole_or_not_at_all() {
        let pc = PrefixCache::new(true, 8, 4, 4, 8192, 0);
        let kv_store = KvCacheStore::new(std::time::Duration::from_secs(600));
        make_recurrent_kv(&kv_store, "turn1", 12, 12);
        let tokens: Vec<u32> = (1..=12).collect();
        let announced = pc.insert_from_kv(
            "m",
            "turn1",
            &kv_store,
            &tokens,
            ANYWHERE,
            KeepAsked::Unasked,
        );
        assert_eq!(pc.entry_count("m"), 1, "the recurrent cache was not saved");
        assert!(announced.is_empty(), "a recurrent snapshot was announced");
        assert!(pc.enumerate_manifest("m").is_empty());
        for block in compute_block_hashes(&tokens, 4) {
            assert!(pc.export_snapshot_bytes("m", &block.block_hash).is_none());
        }

        // The next turn repeats it all: restored whole, state included.
        let next: Vec<u32> = (1..=20).collect();
        let snap = pc.lookup("m", &next).expect("the next turn missed");
        assert_eq!(snap.token_count, 12);
        assert!(snap.carries_recurrent_state());
        assert!(
            serialize_snapshot_with_block_size(&snap, &next, Some(4)).is_err(),
            "a recurrent snapshot was put on the wire"
        );

        // A prompt sharing only 10 tokens is a miss — the state cannot be
        // cut back to them — as is the identical prompt, which would leave
        // no token to read.
        let branch: Vec<u32> = (1..=10).chain([99, 98, 97, 96]).collect();
        assert!(pc.lookup("m", &branch).is_none());
        assert!(pc.lookup("m", &tokens).is_none());

        // The control: an attention-only cache serves that branch narrowed.
        let plain = PrefixCache::new(true, 8, 4, 4, 8192, 0);
        make_fake_kv(&kv_store, "m", "plain", 2, 12);
        plain.insert_from_kv(
            "m",
            "plain",
            &kv_store,
            &tokens,
            ANYWHERE,
            KeepAsked::Unasked,
        );
        assert_eq!(plain.lookup("m", &branch).map(|s| s.token_count), Some(10));
    }

    /// The state is saved only where the attention layers are: one taken a
    /// position past them, or attention narrowed to a shorter opening beside
    /// a whole state, would restore two different histories.
    #[test]
    fn a_recurrent_state_is_saved_only_at_the_attention_layers_position() {
        let pc = PrefixCache::new(true, 8, 4, 4, 8192, 0);
        let kv_store = KvCacheStore::new(std::time::Duration::from_secs(600));
        let tokens: Vec<u32> = (1..=12).collect();
        make_recurrent_kv(&kv_store, "ahead", 12, 13);
        pc.insert_from_kv(
            "m",
            "ahead",
            &kv_store,
            &tokens,
            ANYWHERE,
            KeepAsked::Unasked,
        );
        make_recurrent_kv(&kv_store, "shorter", 12, 12);
        pc.insert_from_kv(
            "m",
            "shorter",
            &kv_store,
            &tokens[..10],
            ANYWHERE,
            KeepAsked::Unasked,
        );
        assert_eq!(pc.entry_count("m"), 0);
        // The control: the same cache saved at its own length is kept.
        pc.insert_from_kv(
            "m",
            "shorter",
            &kv_store,
            &tokens,
            ANYWHERE,
            KeepAsked::Unasked,
        );
        assert_eq!(pc.entry_count("m"), 1);
    }

    /// A card's step writes into the state pair it read two steps before
    /// (`SsmState`), so the snapshot copies the live state and a restore
    /// copies the snapshot: neither request can rewrite what the next hit
    /// restores. The snapshot is charged for the state too.
    #[test]
    fn a_saved_recurrent_state_is_a_copy_on_both_sides() {
        let pc = PrefixCache::new(true, 8, 4, 4, 8192, 0);
        let kv_store = KvCacheStore::new(std::time::Duration::from_secs(600));
        let tokens: Vec<u32> = (1..=12).collect();
        make_recurrent_kv(&kv_store, "turn1", 12, 12);
        pc.insert_from_kv(
            "m",
            "turn1",
            &kv_store,
            &tokens,
            ANYWHERE,
            KeepAsked::Unasked,
        );
        let key = KvCacheStore::cache_key("m", "turn1");
        let saved = recurrent_values(
            kv_store.get_entry(&key).unwrap().ssm_states[1]
                .as_ref()
                .unwrap(),
        );
        let overwrite = |state: &SsmState| {
            let s = &state.current().recurrent_state;
            s.slice_set(&s.zeros_like().unwrap(), 0, 0).unwrap();
        };
        overwrite(
            kv_store.get_entry(&key).unwrap().ssm_states[1]
                .as_ref()
                .unwrap(),
        );

        let next: Vec<u32> = (1..=20).collect();
        let snap = pc.lookup("m", &next).unwrap();
        assert_eq!(snap.bytes(), 12 * 4 * 4 * 2 + (6 + 8) * 4);
        let restored = pc
            .hydrate_request_from_snapshot(&kv_store, "m", "turn2", &snap, &[])
            .unwrap();
        assert_eq!(restored, 12);
        let key2 = KvCacheStore::cache_key("m", "turn2");
        {
            let entry = kv_store.get_entry(&key2).unwrap();
            let state = entry.ssm_states[1].as_ref().unwrap();
            assert_eq!(state.positions(), 12);
            assert_eq!(
                recurrent_values(state),
                saved,
                "the saved state followed the live one"
            );
            overwrite(state);
        }
        let again = pc.lookup("m", &next).unwrap();
        pc.hydrate_request_from_snapshot(&kv_store, "m", "turn3", &again, &[])
            .unwrap();
        let entry = kv_store
            .get_entry(&KvCacheStore::cache_key("m", "turn3"))
            .unwrap();
        assert_eq!(
            recurrent_values(entry.ssm_states[1].as_ref().unwrap()),
            saved,
            "a restored request rewrote the snapshot"
        );
    }

    #[test]
    fn export_snapshot_bytes_roundtrips_to_hashed_block() {
        // block=4 so a 12-token prompt inserts hashes at 4, 8, 12.
        let pc = PrefixCache::new(true, 8, 4, 4, 8192, 0);
        let kv_store = KvCacheStore::new(std::time::Duration::from_secs(600));
        make_fake_kv(&kv_store, "m", "req-a", 2, 12);
        let tokens: Vec<u32> = (1..=12).collect();
        let manifest = pc.insert_from_kv(
            "m",
            "req-a",
            &kv_store,
            &tokens,
            ANYWHERE,
            KeepAsked::Unasked,
        );
        assert_eq!(manifest.len(), 3);
        // Request the middle block. Serving side should produce bytes
        // whose header `tokens` equal tokens[..8].
        let bytes = pc
            .export_snapshot_bytes("m", &manifest[1].block_hash)
            .expect("hit");
        let (snap, decoded_tokens) =
            deserialize_snapshot(&bytes, &candle_core::Device::Cpu).unwrap();
        assert_eq!(snap.token_count, 8);
        assert_eq!(decoded_tokens, tokens[..8]);
        // Same bytes should BLAKE3-verify against the requested hash.
        assert!(verify_token_hash_chain(
            &decoded_tokens,
            4,
            snap.token_count,
            &manifest[1].block_hash
        ));
    }

    #[test]
    fn snapshot_is_finite_accepts_plain_tensors() {
        use candle_core::Device;
        let device = Device::Cpu;
        let k = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (1usize, 1, 2, 2), &device).unwrap();
        let v = Tensor::from_vec(vec![5.0f32, 6.0, 7.0, 8.0], (1usize, 1, 2, 2), &device).unwrap();
        let snap = KvSnapshot {
            token_count: 2,
            layers: vec![Some((k, v))],
            dim: 2,
            max_seq_len: 128,
            restore_to: Vec::new(),
            held_from: Vec::new(),
            serves_from: 0,
            ssm: Vec::new(),
        };
        assert!(snapshot_is_finite(&snap));
    }

    #[test]
    fn snapshot_is_finite_rejects_nan() {
        use candle_core::Device;
        let device = Device::Cpu;
        let k =
            Tensor::from_vec(vec![1.0f32, f32::NAN, 3.0, 4.0], (1usize, 1, 2, 2), &device).unwrap();
        let v = Tensor::from_vec(vec![5.0f32, 6.0, 7.0, 8.0], (1usize, 1, 2, 2), &device).unwrap();
        let snap = KvSnapshot {
            token_count: 2,
            layers: vec![Some((k, v))],
            dim: 2,
            max_seq_len: 128,
            restore_to: Vec::new(),
            held_from: Vec::new(),
            serves_from: 0,
            ssm: Vec::new(),
        };
        assert!(!snapshot_is_finite(&snap));
    }

    #[test]
    fn snapshot_is_finite_rejects_inf() {
        use candle_core::Device;
        let device = Device::Cpu;
        let k = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (1usize, 1, 2, 2), &device).unwrap();
        let v = Tensor::from_vec(
            vec![5.0f32, 6.0, f32::INFINITY, 8.0],
            (1usize, 1, 2, 2),
            &device,
        )
        .unwrap();
        let snap = KvSnapshot {
            token_count: 2,
            layers: vec![Some((k, v))],
            dim: 2,
            max_seq_len: 128,
            restore_to: Vec::new(),
            held_from: Vec::new(),
            serves_from: 0,
            ssm: Vec::new(),
        };
        assert!(!snapshot_is_finite(&snap));
    }

    #[test]
    fn snapshot_is_finite_ignores_none_layers() {
        let snap = KvSnapshot {
            token_count: 0,
            layers: vec![None, None],
            dim: 2,
            max_seq_len: 128,
            restore_to: Vec::new(),
            held_from: Vec::new(),
            serves_from: 0,
            ssm: Vec::new(),
        };
        assert!(snapshot_is_finite(&snap));
    }

    #[test]
    fn export_snapshot_bytes_miss_on_unknown_hash() {
        let pc = PrefixCache::new(true, 8, 4, 4, 8192, 0);
        let kv_store = KvCacheStore::new(std::time::Duration::from_secs(600));
        make_fake_kv(&kv_store, "m", "req-a", 2, 8);
        let tokens: Vec<u32> = (1..=8).collect();
        let _ = pc.insert_from_kv(
            "m",
            "req-a",
            &kv_store,
            &tokens,
            ANYWHERE,
            KeepAsked::Unasked,
        );
        // Hash derived from a different prompt should miss.
        let other = compute_block_hashes(&[99, 99, 99, 99, 99, 99, 99, 99], 4);
        assert!(pc
            .export_snapshot_bytes("m", &other[0].block_hash)
            .is_none());
    }

    #[test]
    fn snapshot_preserves_none_layers() {
        use candle_core::Device;
        let device = Device::Cpu;
        let k = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (1usize, 1, 2, 2), &device).unwrap();
        let v = Tensor::from_vec(vec![5.0f32, 6.0, 7.0, 8.0], (1usize, 1, 2, 2), &device).unwrap();
        let snap = KvSnapshot {
            token_count: 2,
            layers: vec![None, Some((k, v)), None],
            dim: 2,
            max_seq_len: 128,
            restore_to: Vec::new(),
            held_from: Vec::new(),
            serves_from: 0,
            ssm: Vec::new(),
        };
        let tokens: Vec<u32> = vec![11, 12];
        let bytes = serialize_snapshot_with_block_size(&snap, &tokens, None).unwrap();
        let (decoded, _) = deserialize_snapshot(&bytes, &device).unwrap();
        assert_eq!(decoded.layers.len(), 3);
        assert!(decoded.layers[0].is_none());
        assert!(decoded.layers[1].is_some());
        assert!(decoded.layers[2].is_none());
    }

    #[test]
    fn lru_eviction_drops_oldest() {
        let pc = PrefixCache::new(true, 2, 0, 4, 8192, 0);
        let kv_store = KvCacheStore::new(std::time::Duration::from_secs(600));

        for (i, req) in ["r1", "r2", "r3"].iter().enumerate() {
            make_fake_kv(&kv_store, "m", req, 2, 10);
            let tokens: Vec<u32> = (i as u32 * 10..i as u32 * 10 + 10).collect();
            pc.insert_from_kv("m", req, &kv_store, &tokens, ANYWHERE, KeepAsked::Unasked);
        }

        assert_eq!(pc.entry_count("m"), 2);
    }

    /// The bound that actually expresses memory. Counting entries says nothing
    /// about how much is held, because an entry is sized by its prompt: at the
    /// insert ceiling on a 3B model one can reach about 1.9 GB, so sixteen
    /// distinct long conversations could retain about 30 GB while never
    /// exceeding a cap of sixteen.
    #[test]
    fn the_byte_budget_evicts_even_when_the_entry_count_is_fine() {
        // Room for plenty of entries, but only enough bytes for a couple.
        let kv_store = KvCacheStore::new(std::time::Duration::from_secs(600));
        make_fake_kv(&kv_store, "m", "probe", 2, 10);
        let one = {
            let pc = PrefixCache::new(true, 64, 0, 4, 8192, 0);
            pc.insert_from_kv(
                "m",
                "probe",
                &kv_store,
                &(0u32..10).collect::<Vec<_>>(),
                ANYWHERE,
                KeepAsked::Unasked,
            );
            pc.bytes_held("m")
        };
        assert!(one > 0, "a snapshot must weigh something");

        let pc = PrefixCache::new(true, 64, 0, 4, 8192, one * 2);
        for (i, req) in ["r1", "r2", "r3", "r4", "r5"].iter().enumerate() {
            make_fake_kv(&kv_store, "m", req, 2, 10);
            let tokens: Vec<u32> = (i as u32 * 10..i as u32 * 10 + 10).collect();
            pc.insert_from_kv("m", req, &kv_store, &tokens, ANYWHERE, KeepAsked::Unasked);
        }
        assert!(
            pc.entry_count("m") < 5,
            "the byte budget must evict even though the entry cap was never reached"
        );
        assert!(
            pc.bytes_held("m") <= one * 2,
            "held bytes must stay within budget, got {}",
            pc.bytes_held("m")
        );
    }

    /// `make_fake_kv`'s weight per position: two layers, K and V, four f32
    /// each.
    const FAKE_BYTES_PER_POSITION: usize = 2 * 2 * 4 * 4;

    /// A budget smaller than one whole prompt keeps the OPENING that fits,
    /// never nothing and never more than the budget. It used to snapshot the
    /// whole prompt and keep it over budget, because the only alternative was
    /// discarding a copy already paid for; sizing the copy first removes that
    /// choice.
    #[test]
    fn a_budget_smaller_than_one_prompt_keeps_the_opening_that_fits() {
        let kv_store = KvCacheStore::new(std::time::Duration::from_secs(600));
        let pc = PrefixCache::new(true, 64, 0, 4, 8192, FAKE_BYTES_PER_POSITION * 5);
        make_fake_kv(&kv_store, "m", "r1", 2, 10);
        pc.insert_from_kv(
            "m",
            "r1",
            &kv_store,
            &(0u32..10).collect::<Vec<_>>(),
            ANYWHERE,
            KeepAsked::Unasked,
        );
        assert_eq!(pc.entry_count("m"), 1);
        assert!(pc.bytes_held("m") <= FAKE_BYTES_PER_POSITION * 5);
        let hit = pc.lookup("m", &(0u32..20).collect::<Vec<_>>()).unwrap();
        assert_eq!(hit.token_count, 5, "the five positions the budget holds");
    }

    /// A prompt longer than the token ceiling keeps its OPENING — the system
    /// prompt and tool definitions every later turn repeats. It was refused
    /// outright, so by default no agent prompt over 8,192 tokens was ever
    /// cached (field report 2026-09-26: harnesses send 4,100-14,400).
    #[test]
    fn a_prompt_over_the_token_ceiling_keeps_its_opening() {
        let kv_store = KvCacheStore::new(std::time::Duration::from_secs(600));
        let pc = PrefixCache::new(true, 8, 0, 4, 6, 0);
        make_fake_kv(&kv_store, "m", "r1", 2, 10);
        pc.insert_from_kv(
            "m",
            "r1",
            &kv_store,
            &(0u32..10).collect::<Vec<_>>(),
            ANYWHERE,
            KeepAsked::Unasked,
        );
        assert_eq!(pc.entry_count("m"), 1, "the opening is kept, not refused");
        let later: Vec<u32> = (0u32..10).chain(20..30).collect();
        let hit = pc
            .lookup("m", &later)
            .expect("a later turn finds the opening");
        assert_eq!(hit.token_count, 6);
    }

    /// With no token ceiling (0, the default) the byte budget alone decides
    /// how much one snapshot keeps, from what a position of THIS model weighs —
    /// and with neither bound the whole prompt is kept.
    #[test]
    fn with_no_token_ceiling_the_byte_budget_sizes_the_snapshot() {
        let kv_store = KvCacheStore::new(std::time::Duration::from_secs(600));
        let tokens: Vec<u32> = (0u32..10).collect();
        let probe: Vec<u32> = (0u32..20).collect();

        let pc = PrefixCache::new(true, 8, 0, 4, 0, FAKE_BYTES_PER_POSITION * 7);
        make_fake_kv(&kv_store, "m", "r1", 2, 10);
        pc.insert_from_kv("m", "r1", &kv_store, &tokens, ANYWHERE, KeepAsked::Unasked);
        assert_eq!(pc.lookup("m", &probe).unwrap().token_count, 7);
        assert_eq!(pc.bytes_held("m"), FAKE_BYTES_PER_POSITION * 7);

        let unbounded = PrefixCache::new(true, 8, 0, 4, 0, 0);
        make_fake_kv(&kv_store, "m", "r2", 2, 10);
        unbounded.insert_from_kv("m", "r2", &kv_store, &tokens, ANYWHERE, KeepAsked::Unasked);
        assert_eq!(unbounded.lookup("m", &probe).unwrap().token_count, 10);
    }

    /// A zero budget means no byte bound, which is what every existing caller
    /// and every other test relies on.
    #[test]
    fn a_zero_budget_is_unbounded() {
        let kv_store = KvCacheStore::new(std::time::Duration::from_secs(600));
        let pc = PrefixCache::new(true, 64, 0, 4, 8192, 0);
        for (i, req) in ["r1", "r2", "r3", "r4"].iter().enumerate() {
            make_fake_kv(&kv_store, "m", req, 2, 10);
            let tokens: Vec<u32> = (i as u32 * 10..i as u32 * 10 + 10).collect();
            pc.insert_from_kv("m", req, &kv_store, &tokens, ANYWHERE, KeepAsked::Unasked);
        }
        assert_eq!(pc.entry_count("m"), 4);
    }

    // ---- A sliding-window layer's cache keeps its window (FUTURE_WORK #281) ----

    /// A two-layer conversation of `positions`, written as a forward would in
    /// 7-position pieces of a 128-position context: layer 0 windowed (4), so its
    /// cache slides past 128, layer 1 global. Every value is its position.
    fn make_windowed_kv(kv_store: &KvCacheStore, request_id: &str, positions: usize) {
        use crate::inference::layers::append_to_cache;
        use crate::inference::split::kv_cache::KvStorage;
        let mut entry = kv_store.get_or_create("m", request_id, 2);
        let mut pos = 0;
        while pos < positions {
            let len = 7.min(positions - pos);
            let k = Tensor::arange(pos as f32, (pos + len) as f32, &Device::Cpu)
                .unwrap()
                .reshape((1, 1, len, 1))
                .unwrap()
                .broadcast_as((1, 1, len, 4))
                .unwrap()
                .contiguous()
                .unwrap();
            for (slot, window) in entry.layers.iter_mut().zip([Some(4), None]) {
                append_to_cache(slot, &k, &k, pos, KvStorage::F32, 128, 0, window).unwrap();
            }
            pos += len;
        }
    }

    /// The positions layer `i` of a request's cache holds, read back.
    fn positions_in(kv_store: &KvCacheStore, request_id: &str, i: usize) -> Vec<usize> {
        let key = KvCacheStore::cache_key("m", request_id);
        let entry = kv_store.get_entry(&key).unwrap();
        let k = entry.layers[i].as_ref().unwrap().k().unwrap().unwrap();
        k.narrow(3, 0, 1)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
            .into_iter()
            .map(|p| p as usize)
            .collect()
    }

    /// A slid conversation's snapshot carries where each layer starts, is
    /// restored there, and is found only for a prefix its window still
    /// answers; it is never announced to peers or written to the wire.
    #[test]
    fn a_slid_snapshot_is_restored_where_its_window_starts_and_stays_on_this_node() {
        let kv_store = KvCacheStore::new(std::time::Duration::from_secs(600));
        let pc = PrefixCache::new(true, 8, 16, 4, 0, 0);
        make_windowed_kv(&kv_store, "r", 400);
        let held = positions_in(&kv_store, "r", 0);
        let from = held[0];
        assert!(from > 0, "the windowed layer slid");
        let tokens: Vec<u32> = (0..400).collect();
        let manifest =
            pc.insert_from_kv("m", "r", &kv_store, &tokens, ANYWHERE, KeepAsked::Unasked);
        assert!(manifest.is_empty(), "a slid snapshot is not announced");
        assert!(pc.enumerate_manifest("m").is_empty());

        let mut probe = tokens.clone();
        probe.push(9999);
        let snap = pc.lookup("m", &probe).expect("the whole prompt hits");
        assert_eq!(snap.token_count, 400);
        assert_eq!(snap.held_from, [from, 0]);
        assert_eq!(snap.serves_from, from + 3);
        assert!(serialize_snapshot_with_block_size(&snap, &tokens, Some(16)).is_err());

        pc.hydrate_request_from_snapshot(&kv_store, "m", "restored", &snap, &[])
            .unwrap();
        assert_eq!(positions_in(&kv_store, "restored", 0), held);
        assert_eq!(
            positions_in(&kv_store, "restored", 1),
            (0..400).collect::<Vec<_>>()
        );
        let key = KvCacheStore::cache_key("m", "restored");
        let entry = kv_store.get_entry(&key).unwrap();
        let restored = entry.layers[0].as_ref().unwrap();
        assert_eq!(restored.held_from(), from);
        assert_eq!(restored.current_seq_len(), 400);
        drop(entry);

        // A prompt sharing less than the window still answers misses — the
        // windowed layer no longer holds that prefix's keys.
        let mut short: Vec<u32> = (0..200).collect();
        short.push(9999);
        assert!(pc.lookup("m", &short).is_none());

        // The control: an unslid conversation's snapshot is announced.
        let unslid = PrefixCache::new(true, 8, 16, 4, 0, 0);
        make_windowed_kv(&kv_store, "short", 100);
        let tokens: Vec<u32> = (0..100).collect();
        assert!(!unslid
            .insert_from_kv(
                "m",
                "short",
                &kv_store,
                &tokens,
                ANYWHERE,
                KeepAsked::Unasked
            )
            .is_empty());
    }

    /// A long conversation's snapshot taken before its windowed layer ever
    /// slid — a peer's, or one kept with sliding off — is restored to that
    /// layer as its first slide would leave it: only what the window reads, in
    /// a buffer its window and one quantum wide, never the whole prompt, which
    /// every estimate of its cost says it does not hold (FUTURE_WORK #281).
    #[test]
    fn a_long_snapshot_is_restored_to_a_windowed_layer_as_its_window() {
        use crate::inference::layers::{append_to_cache, KV_CACHE_GROWTH_TOKENS};
        use crate::inference::split::kv_cache::{
            positions_kept_for_window, KvStorage, NO_SLIDE_IN_TEST,
        };
        let kv_store = KvCacheStore::new(std::time::Duration::from_secs(600));
        let pc = PrefixCache::new(true, 8, 16, 4, 0, 0);
        let positions = 2000;
        NO_SLIDE_IN_TEST.with(|off| off.set(true));
        {
            let mut entry = kv_store.get_or_create("m", "r", 2);
            for pos in (0..positions).step_by(500) {
                let k = Tensor::arange(pos as f32, (pos + 500) as f32, &Device::Cpu)
                    .unwrap()
                    .reshape((1, 1, 500, 1))
                    .unwrap()
                    .broadcast_as((1, 1, 500, 4))
                    .unwrap()
                    .contiguous()
                    .unwrap();
                for (slot, window) in entry.layers.iter_mut().zip([Some(4), None]) {
                    append_to_cache(slot, &k, &k, pos, KvStorage::F32, 4096, 0, window).unwrap();
                }
            }
        }
        NO_SLIDE_IN_TEST.with(|off| off.set(false));
        let tokens: Vec<u32> = (0..positions as u32).collect();
        assert!(
            !pc.insert_from_kv("m", "r", &kv_store, &tokens, ANYWHERE, KeepAsked::Unasked)
                .is_empty(),
            "a snapshot that never slid is announced"
        );
        let mut probe = tokens.clone();
        probe.push(9999);
        let snap = pc.lookup("m", &probe).expect("the whole prompt hits");
        let layouts = [
            Some(CacheLayout {
                storage: KvStorage::F32,
                window: Some(4),
            }),
            Some(CacheLayout {
                storage: KvStorage::F32,
                window: None,
            }),
        ];
        pc.hydrate_request_from_snapshot(&kv_store, "m", "restored", &snap, &layouts)
            .unwrap();
        let keep = positions_kept_for_window(4);
        assert_eq!(
            positions_in(&kv_store, "restored", 0),
            (positions - keep..positions).collect::<Vec<_>>()
        );
        assert_eq!(
            positions_in(&kv_store, "restored", 1),
            (0..positions).collect::<Vec<_>>()
        );
        let key = KvCacheStore::cache_key("m", "restored");
        let entry = kv_store.get_entry(&key).unwrap();
        let windowed = entry.layers[0].as_ref().unwrap();
        assert_eq!(windowed.current_seq_len(), positions);
        assert_eq!(windowed.serves_from(), positions - keep + 3);
        let buffer = windowed.k_cache().max_seq_len();
        assert!(
            (keep..=keep + KV_CACHE_GROWTH_TOKENS).contains(&buffer),
            "the windowed layer's buffer is {buffer} positions"
        );
        assert!(entry.layers[1].as_ref().unwrap().k_cache().max_seq_len() >= positions);
        drop(entry);

        // The control: restored with no layouts, the windowed layer is built
        // to the whole prompt, as every restore was before.
        pc.hydrate_request_from_snapshot(&kv_store, "m", "unshaped", &snap, &[])
            .unwrap();
        assert_eq!(
            positions_in(&kv_store, "unshaped", 0),
            (0..positions).collect::<Vec<_>>()
        );
    }

    /// A slid layer weighs per position what it weighed before it slid, and
    /// the positions it no longer holds are counted apart: the snapshot
    /// ceiling is bytes over this figure, so a lighter reading would let a
    /// snapshot past its byte budget.
    #[test]
    fn a_slid_layer_weighs_the_same_per_position() {
        let kv_store = KvCacheStore::new(std::time::Duration::from_secs(600));
        make_windowed_kv(&kv_store, "r", 400);
        let key = KvCacheStore::cache_key("m", "r");
        let entry = kv_store.get_entry(&key).unwrap();
        let held_from = entry.layers[0].as_ref().unwrap().held_from();
        assert!(held_from > 0);
        // Two layers, K and V of [1, 1, ·, 4] f32: 32 bytes a position each.
        assert_eq!(
            SnapshotCost::of(&entry.layers),
            SnapshotCost {
                per_position: 64,
                slid_away: 32 * held_from,
            }
        );
    }

    /// A snapshot copies a slid layer from where it starts (`snapshot_at`), so
    /// the byte budget is spent on those positions alone: a budget of exactly
    /// what the snapshot weighs keeps it whole. Charged every position, the
    /// windowed layer kept a shorter opening — or, where that fell short of
    /// where its window answers, nothing (#281).
    #[test]
    fn a_byte_budget_charges_a_slid_layer_only_the_positions_it_holds() {
        let kv_store = KvCacheStore::new(std::time::Duration::from_secs(600));
        make_windowed_kv(&kv_store, "r", 400);
        let held_from = {
            let key = KvCacheStore::cache_key("m", "r");
            let entry = kv_store.get_entry(&key).unwrap();
            entry.layers[0].as_ref().unwrap().held_from()
        };
        // The whole layer holds 400 positions, the windowed one 400 - held_from.
        let weighs = 32 * 400 + 32 * (400 - held_from);
        let prompt: Vec<u32> = (0..400).collect();
        let mut next_turn = prompt.clone();
        next_turn.push(9999);

        let pc = PrefixCache::new(true, 8, 0, 4, 0, weighs);
        pc.insert_from_kv("m", "r", &kv_store, &prompt, ANYWHERE, KeepAsked::Unasked);
        let hit = pc.lookup("m", &next_turn).expect("kept within its budget");
        assert_eq!(hit.token_count, 400);

        // The budget still binds: a byte less keeps a shorter opening.
        let tight = PrefixCache::new(true, 8, 0, 4, 0, weighs - 1);
        tight.insert_from_kv("m", "r", &kv_store, &prompt, ANYWHERE, KeepAsked::Unasked);
        let hit = tight
            .lookup("m", &next_turn)
            .expect("its window answers 399");
        assert_eq!(hit.token_count, 399);
    }

    /// An earlier turn the slid snapshot cannot serve is kept beside it, and a
    /// prompt branching between the two is served from the earlier one.
    #[test]
    fn an_earlier_turn_a_slid_snapshot_cannot_serve_is_kept_and_found() {
        let kv_store = KvCacheStore::new(std::time::Duration::from_secs(600));
        let pc = PrefixCache::new(true, 8, 0, 4, 0, 0);
        make_windowed_kv(&kv_store, "turn-1", 100);
        let turn_1: Vec<u32> = (0..100).collect();
        pc.insert_from_kv(
            "m",
            "turn-1",
            &kv_store,
            &turn_1,
            ANYWHERE,
            KeepAsked::Unasked,
        );
        make_windowed_kv(&kv_store, "turn-2", 400);
        let turn_2: Vec<u32> = (0..400).collect();
        pc.insert_from_kv(
            "m",
            "turn-2",
            &kv_store,
            &turn_2,
            ANYWHERE,
            KeepAsked::Unasked,
        );
        assert_eq!(pc.entry_count("m"), 2, "turn 1 is not covered by turn 2");

        let mut branch: Vec<u32> = (0..200).collect();
        branch.push(9999);
        let hit = pc.lookup("m", &branch).expect("turn 1 serves the branch");
        assert_eq!(hit.token_count, 100);
        assert!(hit.held_from.iter().all(|&f| f == 0));
    }

    /// A split's stored prompt: kept only at a length its window answers, and
    /// found only for one.
    #[test]
    fn a_slid_split_prompt_is_stored_and_found_only_where_its_window_answers() {
        let kv_store = KvCacheStore::new(std::time::Duration::from_secs(600));
        let pc = PrefixCache::new(true, 8, 0, 4, 0, 0);
        make_windowed_kv(&kv_store, "r", 400);
        let keys: Vec<[u8; 32]> = (0..7u8).map(|i| [i; 32]).collect();
        assert_eq!(
            pc.insert_chain_from_kv("m", "r", &kv_store, 64, &keys[..2], ANYWHERE),
            0,
            "128 positions are short of where the window answers"
        );
        assert_eq!(
            pc.insert_chain_from_kv("m", "r", &kv_store, 64, &keys, ANYWHERE),
            6
        );
        assert!(pc.lookup_chain("m", 64, &keys, 128).is_none());
        let hit = pc.lookup_chain("m", 64, &keys, 384).unwrap();
        assert_eq!(hit.token_count, 384);
        assert!(hit.held_from[0] > 0);
    }
}
