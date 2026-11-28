//! A tree of guesses checked in one pass (#134; `docs/plans/large_model_splits.md`
//! § "Draft trees").
//!
//! A chain's check keeps the run of guesses before its first miss; a tree's
//! keeps the longest run along whichever branch the model takes. Every segment
//! reads the same shape: a row sits at `base + depth` for RoPE and attends the
//! whole confirmed context plus its own ancestors — the tree attention of
//! SpecInfer (arXiv 2305.09781), Medusa and SpecExec (arXiv 2406.02532).
//!
//! A chain is the tree whose every row's parent is the row before it, so the
//! rules here hold for the checks that existed before trees too.

/// The most rows one tree may hold, on the wire (`LayerForward::draft_parents`)
/// and in a worker. Ample: past ~128 rows a pass gains well under a token per
/// doubling (`docs/plans/large_model_splits.md` § "Projections").
pub(crate) const MAX_TREE_ROWS: usize = 512;

/// The u32 words of one row's [`RowVisibility`]: a bit for every row a tree
/// may hold.
pub(crate) const VISIBILITY_WORDS: usize = MAX_TREE_ROWS / 32;

/// Which keys each row of a tree's forward reads: every position before the
/// tree's base, and of the tree's own rows its ancestors and itself — a bit
/// per tree row ([`VISIBILITY_WORDS`] words a forward row). The ONE statement
/// of what a row sees: the card's tree attention reads the bits as they are
/// (`decode_attn::gqa_tree_attention_cuda`, FlashInfer's packed custom mask),
/// and the additive mask every other path reads is made from them
/// ([`Self::mask`]), so the two cannot disagree.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct RowVisibility {
    /// The cache position of the tree's first row.
    pub(crate) base: usize,
    /// The cache's length after the forward: `base` plus the tree's rows.
    pub(crate) kv_len: usize,
    /// `[forward rows, VISIBILITY_WORDS]`, row-major: bit `j % 32` of word
    /// `j / 32` is set where the row reads tree row `j`.
    pub(crate) words: Vec<u32>,
}

impl RowVisibility {
    pub(crate) fn rows(&self) -> usize {
        self.words.len() / VISIBILITY_WORDS
    }

    /// Does forward row `q` read cache position `key`?
    pub(crate) fn sees(&self, q: usize, key: usize) -> bool {
        match key.checked_sub(self.base) {
            None => true,
            Some(j) => self.words[q * VISIBILITY_WORDS + j / 32] >> (j % 32) & 1 == 1,
        }
    }

    /// The row each forward row follows, for a forward holding its WHOLE tree:
    /// the nearest tree row below it that it reads — a row reads exactly its
    /// ancestors and itself (`DraftTree::rows_for`), and its parent comes after
    /// every other ancestor — or `None` for a row that follows the confirmed
    /// context. A layer keeping ONE state a request (Qwen 3.5's recurrent
    /// layers, #437) starts each row from its parent's state by it. Refused for
    /// a forward of only a tree's LAST rows (the drafter grows a tree that
    /// way): such a layer keeps no state for rows an earlier forward read.
    pub(crate) fn parents(&self) -> Result<Vec<Option<usize>>, String> {
        let rows = self.rows();
        if self.kv_len != self.base + rows {
            return Err(format!(
                "a forward of {rows} rows of a tree of {} rows: a recurrent layer reads a tree whole",
                self.kv_len.saturating_sub(self.base)
            ));
        }
        Ok((0..rows)
            .map(|q| (0..q).rev().find(|&j| self.sees(q, self.base + j)))
            .collect())
    }

    /// [`Self::parents`] for a forward of a tree that FLOWS, which may hold
    /// only the tree's LAST rows (`LayerForward::draft_tree_from`): how many
    /// tree rows earlier forwards wrote, and each forward row's parent as a TREE
    /// row — any row before it, this forward's or an earlier one's — or `None`
    /// for one that follows the position before the tree. A recurrent layer
    /// that keeps the inputs of the rows a flow may still need
    /// (`layers::qwen35::FlowRows`) reads them by it.
    pub(crate) fn parents_across_forwards(&self) -> Result<(usize, Vec<Option<usize>>), String> {
        let rows = self.rows();
        let before = self
            .kv_len
            .checked_sub(self.base + rows)
            .ok_or_else(|| format!("a forward of {rows} rows past its tree's end"))?;
        Ok((
            before,
            (0..rows)
                .map(|q| (0..before + q).rev().find(|&j| self.sees(q, self.base + j)))
                .collect(),
        ))
    }

    /// The additive mask `[rows, kv_len]`, row-major: 0 where a row reads a
    /// key, −∞ where it does not — what the matmul path adds to its scores.
    pub(crate) fn mask(&self) -> Vec<f32> {
        let mut mask = vec![f32::NEG_INFINITY; self.rows() * self.kv_len];
        for (q, line) in mask.chunks_mut(self.kv_len).enumerate() {
            for (key, m) in line.iter_mut().enumerate() {
                if self.sees(q, key) {
                    *m = 0.0;
                }
            }
        }
        mask
    }
}

/// The shape of a tree of guesses: for every row, in the order the rows are
/// written to the cache, the row it follows.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DraftTree {
    /// Where the tree's first row sits in the cache: everything before it is
    /// confirmed context, and every row reads all of it.
    base: usize,
    /// Row i's parent among the tree's rows, or `None` for a row that follows
    /// the position just before `base`. A parent always comes before its child.
    parents: Vec<Option<u16>>,
    /// Row i's distance from the confirmed context: 0 for a row with no parent.
    depths: Vec<u32>,
}

impl DraftTree {
    /// A tree from its parents, refused when a row names a parent that does not
    /// come before it — the shape a cache written in row order could not hold —
    /// or when it is past [`MAX_TREE_ROWS`].
    pub fn new(base: usize, parents: Vec<Option<u16>>) -> Result<Self, String> {
        // A forward at position 0 empties the cache it writes
        // (`layers::append_to_cache`), so a tree always follows something.
        if base == 0 {
            return Err("a tree of guesses must follow at least one position".into());
        }
        if parents.len() > MAX_TREE_ROWS {
            return Err(format!(
                "a tree of {} rows is past the {MAX_TREE_ROWS} one check may hold",
                parents.len()
            ));
        }
        let mut depths: Vec<u32> = Vec::with_capacity(parents.len());
        for (i, parent) in parents.iter().enumerate() {
            let depth = match parent {
                None => 0,
                Some(p) if (*p as usize) < i => depths[*p as usize] + 1,
                Some(p) => {
                    return Err(format!("row {i} of a tree names row {p} as its parent"));
                }
            };
            depths.push(depth);
        }
        Ok(Self {
            base,
            parents,
            depths,
        })
    }

    /// The wire form (`LayerForward::draft_parents`): 0 for a row that follows
    /// the position before the forward, `k` for one whose parent is row `k - 1`.
    pub(crate) fn from_wire(base: usize, wire: &[u16]) -> Result<Self, String> {
        Self::new(base, wire.iter().map(|&w| w.checked_sub(1)).collect())
    }

    pub(crate) fn to_wire(&self) -> Vec<u16> {
        self.parents
            .iter()
            .map(|p| p.map_or(0, |p| p + 1))
            .collect()
    }

    pub(crate) fn len(&self) -> usize {
        self.parents.len()
    }

    /// Where the tree's first row sits in the cache.
    pub(crate) fn base(&self) -> usize {
        self.base
    }

    pub(crate) fn parent(&self, row: usize) -> Option<usize> {
        self.parents[row].map(usize::from)
    }

    pub(crate) fn depth(&self, row: usize) -> usize {
        self.depths[row] as usize
    }

    /// Add a row following `parent` (a row already in the tree, or `None`),
    /// returning its index — how the drafter grows its tree a step at a time.
    pub(crate) fn push(&mut self, parent: Option<usize>) -> Result<usize, String> {
        let row = self.parents.len();
        if row >= MAX_TREE_ROWS {
            return Err(format!("a tree may hold {MAX_TREE_ROWS} rows"));
        }
        let depth = match parent {
            None => 0,
            Some(p) if p < row => self.depths[p] + 1,
            Some(p) => return Err(format!("row {row} of a tree names row {p} as its parent")),
        };
        self.parents.push(parent.map(|p| p as u16));
        self.depths.push(depth);
        Ok(row)
    }

    /// Is every row's parent the row before it — the shape of a chain? Such a
    /// tree reads exactly what an ordinary forward reads (its mask is the
    /// causal one, its positions consecutive), so a worker runs it as one.
    pub(crate) fn is_chain(&self) -> bool {
        self.parents
            .iter()
            .enumerate()
            .all(|(i, p)| *p == i.checked_sub(1).map(|q| q as u16))
    }

    /// The rows `row` reads within the tree: itself and every ancestor.
    fn ancestors_and_self(&self, row: usize) -> impl Iterator<Item = usize> + '_ {
        std::iter::successors(Some(row), move |&r| self.parent(r))
    }

    /// The positions and the keys each row reads ([`RowVisibility`]; its
    /// [`RowVisibility::mask`] is the additive mask) of a forward of `seq_len`
    /// rows written after `kv_offset` cached ones — the tree's LAST `seq_len`
    /// rows, those before them already in the cache from an earlier forward of
    /// the same tree (the drafter grows its tree that way).
    pub(crate) fn rows_for(
        &self,
        kv_offset: usize,
        seq_len: usize,
    ) -> Result<(Vec<u32>, RowVisibility), String> {
        let before = kv_offset.checked_sub(self.base).ok_or_else(|| {
            format!(
                "a tree from position {} read after only {kv_offset} cached positions",
                self.base
            )
        })?;
        if before + seq_len != self.len() {
            return Err(format!(
                "a forward of {seq_len} rows after {before} of a tree of {} rows",
                self.len()
            ));
        }
        let mut words = vec![0u32; seq_len * VISIBILITY_WORDS];
        let mut positions = Vec::with_capacity(seq_len);
        for (q, line) in words.chunks_mut(VISIBILITY_WORDS).enumerate() {
            let row = before + q;
            positions.push((self.base + self.depth(row)) as u32);
            for seen in self.ancestors_and_self(row) {
                line[seen / 32] |= 1 << (seen % 32);
            }
        }
        Ok((
            positions,
            RowVisibility {
                base: self.base,
                kv_len: kv_offset + seq_len,
                words,
            },
        ))
    }

    /// The rows that follow `row` (`None`: the rows with no parent).
    pub(crate) fn children(&self, row: Option<usize>) -> impl Iterator<Item = usize> + '_ {
        let want = row.map(|r| r as u16);
        self.parents
            .iter()
            .enumerate()
            .filter(move |(_, p)| **p == want)
            .map(|(i, _)| i)
    }
}

/// What a tree check's walk kept, read back against the tree that was sent:
/// the guessed rows along the kept path, in order.
///
/// `guessed_from` is the first guessed row — rows before it are tokens the
/// coordinator already holds (the kept tokens a forward re-reads, then the
/// check's own last token), so the walk starts at the row before it. `kept` is
/// the walk's tokens less its last (the sample where it stopped). `None` when
/// the walk names a token no child carries: no honest tail sends that.
pub(crate) fn kept_rows(
    tree: &DraftTree,
    guessed_from: usize,
    tokens: &[u32],
    kept: &[u32],
) -> Option<Vec<usize>> {
    let mut at = guessed_from.checked_sub(1)?;
    let mut rows = Vec::with_capacity(kept.len());
    for &t in kept {
        let next = tree
            .children(Some(at))
            .find(|&c| c >= guessed_from && tokens.get(c) == Some(&t))?;
        rows.push(next);
        at = next;
    }
    Some(rows)
}

/// Where a walk over a tree that flows stopped ([`walk_draws`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum WalkEnd {
    /// On `row`, whose draw has not come back yet: the walk goes on from
    /// there when it does.
    Waiting { row: usize },
    /// The draw at `row` was `token`, which no row following it carries. The
    /// token is emitted; the tree goes on from a new row carrying it.
    Missed { row: usize, token: u32 },
}

/// The requester's walk over a tree that FLOWS (#134 (b)): standing on `at`,
/// take its draw (`draws[at]`, the last segment's sample after that row's
/// path, `sampling::draw_each_row`); keep the row following it that carries
/// the draw and go on, or stop with a draw no row carries. Returns the tokens
/// emitted, in order, and where it stopped.
///
/// This is [`crate::inference::sampling::walk_verified_tree`]'s rule, walked
/// where the tree grows instead of where the logits are: every emitted token
/// is the model's own draw along the path emitted, whichever rows were sent
/// and whenever their draws came back. `tokens[r]` is row r's token; the rows
/// following one row carry distinct tokens, which is what makes "the row
/// carrying the draw" one row.
pub(crate) fn walk_draws(
    tree: &DraftTree,
    tokens: &[u32],
    draws: &[Option<u32>],
    mut at: usize,
) -> (Vec<u32>, WalkEnd) {
    let mut emitted = Vec::new();
    loop {
        let Some(draw) = draws.get(at).copied().flatten() else {
            return (emitted, WalkEnd::Waiting { row: at });
        };
        emitted.push(draw);
        match tree
            .children(Some(at))
            .find(|&c| tokens.get(c) == Some(&draw))
        {
            Some(next) => at = next,
            None => {
                return (
                    emitted,
                    WalkEnd::Missed {
                        row: at,
                        token: draw,
                    },
                )
            }
        }
    }
}

/// Of a forward's `rows`, those a FAR segment still has to compute (#510):
/// the rows the walk from `anchor` — the requester's walk row as it sent the
/// forward — may still reach over the far's own draws ([`walk_draws`]): the
/// row it waits on and every row following it. A walk that missed reaches
/// none: the requester rebases, and a rebase's forward is computed whole.
/// [`FlowTree::is_live`] is the same rule over the draws the requester holds.
///
/// Measured 2026-11-27 (TH→AU, the 14B, the 1.5B helper): 86-90% of the rows
/// a far computed were already off the walk when it served them.
pub(crate) fn rows_the_walk_may_reach(
    tree: &DraftTree,
    tokens: &[u32],
    draws: &[Option<u32>],
    anchor: usize,
    rows: std::ops::Range<usize>,
) -> Vec<usize> {
    let WalkEnd::Waiting { row: at } = walk_draws(tree, tokens, draws, anchor).1 else {
        return Vec::new();
    };
    rows.filter(|&row| row < tree.len() && follows_or_is(tree, row, at))
        .collect()
}

/// Is `row` the row `at` or one following it?
fn follows_or_is(tree: &DraftTree, row: usize, at: usize) -> bool {
    let floor = tree.depth(at);
    let mut r = row;
    while tree.depth(r) > floor {
        match tree.parent(r) {
            Some(p) => r = p,
            None => return false,
        }
    }
    r == at
}

/// The rows a far forward runs (#510), in row order: none where none is live
/// (the forward is answered at once); where at most `width` are live, those
/// padded to `width` with the forward's other rows in order — ONE row count,
/// so one captured shape (a dense far captures up to
/// `cuda_graph::MAX_POSITIONS` rows; a first shape costs a capture inside a
/// reply) — or every row where the forward is no wider than that; past
/// `width` live, exactly the live rows.
pub(crate) fn rows_to_serve(
    live: &[usize],
    rows: std::ops::Range<usize>,
    width: usize,
) -> Vec<usize> {
    if live.is_empty() {
        return Vec::new();
    }
    if live.len() > width {
        return live.to_vec();
    }
    if rows.len() <= width {
        return rows.collect();
    }
    let mut serve = live.to_vec();
    for row in rows {
        if serve.len() >= width {
            break;
        }
        if !live.contains(&row) {
            serve.push(row);
        }
    }
    serve.sort_unstable();
    serve
}

/// Does this node skip dead rows as a flow's far end, and ask a far end to
/// (#510)? `SWARMLLM_FLOW_LIVE_ROWS=0` turns both off: the A/B switch, read at
/// either end — the far's advertisement (`features::FLOW_LIVE_ROWS`) and the
/// requester's flag (`LayerForward::rows_may_skip`).
pub(crate) fn live_rows_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("SWARMLLM_FLOW_LIVE_ROWS").as_deref() != Ok("0"))
}

/// The one shape a few live rows of a far forward are padded to (#510): the
/// widest a dense far captures as one graph (`cuda_graph::MAX_POSITIONS`), so
/// a level of 1-8 live rows runs as one recorded step whatever its count.
pub(crate) const LIVE_ROWS_WIDTH: usize = crate::inference::cuda_graph::MAX_POSITIONS;

/// A far segment's answer for a row it did not draw (#510): no token has this
/// id, so a requester that agreed to skipped rows
/// (`LayerForward::rows_may_skip`) reads it as "not drawn", never as a word.
pub(crate) const NOT_DRAWN: u32 = u32::MAX;

/// What a FAR segment that skips dead rows (#510) keeps of one request's
/// flowing tree between its forwards — beside the request's cache
/// (`KvCacheEntry::live_rows`), so a truncate, a clear and an expiry carry it.
///
/// Its cache holds only the rows it computed, in row order. A row's draw goes
/// back only where the row is live AND sound — computed after every row on
/// its path — so a draw never comes from a cache missing a row it reads: a
/// requester whose walk differs from this one's meets [`NOT_DRAWN`] and checks
/// in rounds, never a wrong word.
///
/// **The first draw sent for a row stands** ([`SentBefore`]): a rebase's
/// forward reads its kept rows again, and the requester keeps the draw that
/// came in first, so this segment answers and walks by that one too.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct LiveRows {
    /// The tree's first position (`LayerForward::draft_tree_from`).
    from: usize,
    /// Per row of the tree: its place among the rows computed (its cache slot
    /// past `from`), `None` where it was skipped.
    local: Vec<Option<u16>>,
    /// Per row: computed after every row on its path.
    sound: Vec<bool>,
    /// Per row: this segment's draw, kept for the rows it answered.
    draws: Vec<Option<u32>>,
    /// Per row: its parent and token as the last forward carried the tree,
    /// so a row can be named by its tokens from `from` once the rows it was
    /// numbered among are cut ([`Self::truncated_to`]).
    shape: Vec<(Option<u16>, u32)>,
    /// The draws sent for a tree this segment's cache was taken back into —
    /// a rebase cuts to the tree's start and reads the kept rows again.
    sent_before: Option<SentBefore>,
}

/// The draws a far segment sent for rows of a tree, by each row's tokens from
/// the tree's start (`base`), kept across the cut a rebase makes (#510).
///
/// A rebase's forward reads its kept rows again in ONE forward, a different
/// row count from the forwards that first drew them, and a card's sums depend
/// on the row count (batch invariance: Thinking Machines, "Defeating
/// Nondeterminism in LLM Inference", 2025; vLLM's `VLLM_BATCH_INVARIANT`). At
/// a near-tie the greedy draw can come out the other way. The requester keeps
/// the draw that came in first (`dsd_flow::take_row_answer`); a far that
/// walked by the second skipped the rows the requester's walk then reached,
/// which ended the flow (2026-11-28, TH→AU, the 14B: 2 replies of 5, one with
/// no miss). Prior art: LLM-42 keeps a token once decided rather than run
/// batch-invariant kernels; vLLM's batch-invariant mode costs speed. Taken
/// here: never decide a row twice.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct SentBefore {
    base: usize,
    draws: std::collections::HashMap<Vec<u32>, u32>,
}

/// The tokens from a tree's start to `row` and including it, over a parent
/// list that names only earlier rows (`DraftTree::new`'s rule); `None` where
/// one does not.
fn tokens_to_row(
    parent: impl Fn(usize) -> Option<usize>,
    token: impl Fn(usize) -> u32,
    row: usize,
) -> Option<Vec<u32>> {
    let mut path = vec![token(row)];
    let mut r = row;
    while let Some(p) = parent(r) {
        if p >= r {
            return None;
        }
        path.push(token(p));
        r = p;
    }
    path.reverse();
    Some(path)
}

impl SentBefore {
    /// The draws `rows` sent, by path; `None` where it sent none or its shape
    /// does not cover them. A path two rows share is left out: neither is the
    /// row a later forward names by it.
    fn of(rows: &LiveRows) -> Option<Self> {
        if rows.shape.len() != rows.draws.len() {
            return None;
        }
        let mut draws = std::collections::HashMap::new();
        let mut shared = std::collections::HashSet::new();
        for (row, draw) in rows.draws.iter().enumerate() {
            let Some(draw) = *draw else { continue };
            let Some(path) = tokens_to_row(
                |r| rows.shape[r].0.map(usize::from),
                |r| rows.shape[r].1,
                row,
            ) else {
                continue;
            };
            if draws.insert(path.clone(), draw).is_some() {
                shared.insert(path);
            }
        }
        for path in &shared {
            draws.remove(path);
        }
        (!draws.is_empty()).then_some(Self {
            base: rows.from,
            draws,
        })
    }

    /// Per row of `tree` (carrying `tokens`), the draw sent for the row with
    /// its path before; `None` for every row where `tree` starts elsewhere.
    fn for_tree(&self, tree: &DraftTree, tokens: &[u32]) -> Vec<Option<u32>> {
        (0..tree.len())
            .map(|row| {
                if tree.base() != self.base {
                    return None;
                }
                tokens_to_row(|r| tree.parent(r), |r| tokens[r], row)
                    .and_then(|path| self.draws.get(&path).copied())
            })
            .collect()
    }
}

impl LiveRows {
    /// Where the requester's position `pos` sits in this segment's cache:
    /// before the tree, where it is; in it, past the rows computed before it.
    pub(crate) fn local_position(&self, pos: usize) -> usize {
        if pos <= self.from {
            return pos;
        }
        let rows = pos - self.from;
        let known = rows.min(self.local.len());
        self.from + self.local[..known].iter().flatten().count() + (rows - known)
    }

    /// What is left once the requester takes its cache back to `pos`: `None`
    /// where that cuts into the positions before the tree. A cut into the
    /// tree remembers the draws it sent ([`SentBefore`]): a rebase cuts to the
    /// tree's start and its forward reads the kept rows again.
    pub(crate) fn truncated_to(mut self, pos: usize) -> Option<Self> {
        let rows = pos.checked_sub(self.from)?;
        if rows < self.draws.len() {
            self.sent_before = SentBefore::of(&self);
        }
        self.local.truncate(rows);
        self.sound.truncate(rows);
        self.draws.truncate(rows);
        self.shape.truncate(rows);
        Some(self)
    }

    /// The same rows held as a tree from `base`: a rebase's forward runs whole
    /// from the old base — the walk's path, then the rows it kept
    /// (`FlowTree::rebase_forward`) — and the next tree starts where the path
    /// ends, its first rows the kept ones. Only where every row before `base`
    /// was computed, so the cache holds those positions where they say; `None`
    /// where one was skipped or `base` lies before the tree or past what is
    /// held (the rig of 2026-11-28 refused every flow at its first rebase
    /// without this).
    fn rebased_to(mut self, base: usize) -> Option<Self> {
        let off = base.checked_sub(self.from)?;
        if off == 0 {
            return Some(self);
        }
        let shift = u16::try_from(off).ok()?;
        if off > self.local.len() || self.local[..off].iter().any(Option::is_none) {
            return None;
        }
        // Every slot past `off` is at least `shift` (slots count the rows run
        // in order, and every row before `off` was run); checked anyway, so a
        // map that ever breaks that is refused, never wrapped (security
        // review, 2026-11-28).
        self.local = self
            .local
            .split_off(off)
            .into_iter()
            .map(|slot| match slot {
                None => Some(None),
                Some(s) => s.checked_sub(shift).map(Some),
            })
            .collect::<Option<Vec<_>>>()?;
        self.sound = self.sound.split_off(off);
        self.draws = self.draws.split_off(off);
        self.from = base;
        // Named again by the forward that reads them (`plan` sets it); the
        // draws sent before a cut are for that cut's own forward only.
        self.shape.clear();
        self.sent_before = None;
        Some(self)
    }

    /// The rows of a far forward this segment runs (#510), given what it kept
    /// of the tree (`held`), the tree and its tokens as this forward carries
    /// them, the requester's walk row as it sent it (`anchor`), the forward's
    /// row count and the one shape a few live rows are padded to (`width`).
    ///
    /// `held` is read only where it describes this tree up to this forward —
    /// kept of it, or of the tree a rebase ran whole before it
    /// ([`Self::rebased_to`]); a tree's first forward starts afresh, and with
    /// nothing held the earlier rows came in forwards that skipped nothing —
    /// the cache holds them all (the continuity check read it so). Rows kept of
    /// a tree this one does not continue, or a different count of its rows,
    /// are refused: the cache is not what the rows say (security review,
    /// 2026-11-28).
    pub(crate) fn plan(
        held: Option<LiveRows>,
        tree: &DraftTree,
        tokens: &[u32],
        anchor: usize,
        positions: usize,
        width: usize,
    ) -> Result<ServeRows, String> {
        let len = tree.len();
        if tokens.len() != len || positions == 0 || positions > len || anchor >= len {
            return Err(format!(
                "skipping rows: {} tokens and {positions} rows for a tree of {len}, walk at {anchor}",
                tokens.len()
            ));
        }
        let read_before = len - positions;
        // A tree's first forward after a cut into it (a rebase's) is answered
        // with the draws sent before for the rows it reads again.
        let sent_before = match &held {
            Some(held) if read_before == 0 => held.sent_before.clone(),
            _ => None,
        };
        let mut state = match held {
            _ if read_before == 0 => LiveRows {
                from: tree.base(),
                ..LiveRows::default()
            },
            Some(held) => {
                let kept_from = held.from;
                let held = held.rebased_to(tree.base()).ok_or_else(|| {
                    format!(
                        "skipping rows: rows kept of a tree from {kept_from} do not start one from {}",
                        tree.base()
                    )
                })?;
                if held.local.len() != read_before {
                    return Err(format!(
                        "skipping rows: {} rows of this tree kept, {read_before} read before",
                        held.local.len()
                    ));
                }
                held
            }
            None => LiveRows {
                from: tree.base(),
                local: (0..read_before).map(|r| Some(r as u16)).collect(),
                sound: vec![true; read_before],
                draws: vec![None; read_before],
                ..LiveRows::default()
            },
        };
        state.sent_before = None;
        let computed_before = state.local.iter().flatten().count();
        let rows = read_before..len;
        let mut draws = state.draws.clone();
        draws.resize(len, None);
        let live = rows_the_walk_may_reach(tree, tokens, &draws, anchor, rows.clone());
        // Rows read again keep the draws sent for them, for the answer and for
        // every later walk here. `live` stays every row from the requester's
        // walk on: a draw it lost to its own rebase (`stale_answers`) is sent
        // again, the first one.
        if let Some(sent) = sent_before {
            state.draws = sent.for_tree(tree, tokens);
        }
        // A tree's first forward runs whole: a rebase's leads with the path
        // the walk took (`FlowTree::rebase_forward`), rows BEFORE the walk's
        // that every kept row reads, and a reply's first has drawn nothing.
        let serve = if read_before == 0 {
            rows.clone().collect()
        } else {
            rows_to_serve(&live, rows.clone(), width)
        };
        let mut computed = computed_before;
        for row in rows.clone() {
            state.local.push(serve.contains(&row).then(|| {
                computed += 1;
                (computed - 1) as u16
            }));
        }
        // Every computed row's parent among the computed rows — the nearest
        // one on its path where its own was skipped, and then it is unsound.
        let mut parents = Vec::with_capacity(computed);
        for row in 0..len {
            if state.local[row].is_none() {
                if row >= read_before {
                    state.sound.push(false);
                }
                continue;
            }
            let mut parent = tree.parent(row);
            let mut whole = true;
            while let Some(p) = parent {
                if state.local[p].is_some() {
                    break;
                }
                whole = false;
                parent = tree.parent(p);
            }
            parents.push(parent.and_then(|p| state.local[p]));
            if row >= read_before {
                let sound = whole && parent.is_none_or(|p| state.sound[p]);
                state.sound.push(sound);
            }
        }
        let answered = serve
            .iter()
            .map(|&row| live.contains(&row) && state.sound[row])
            .collect();
        let compact = if serve.is_empty() {
            None
        } else {
            Some(DraftTree::new(tree.base(), parents)?)
        };
        state.draws.resize(len, None);
        state.shape = (0..len)
            .map(|row| (tree.parent(row).map(|p| p as u16), tokens[row]))
            .collect();
        Ok(ServeRows {
            rows: serve.iter().map(|&row| row - read_before).collect(),
            tree: compact,
            read_before: computed_before,
            answered,
            positions,
            next: state,
        })
    }
}

/// A far forward's rows as its segment runs them ([`LiveRows::plan`]).
#[derive(Debug)]
pub(crate) struct ServeRows {
    /// The forward's rows to run, counted from its first, in order.
    pub(crate) rows: Vec<usize>,
    /// The tree over every row computed, these included; `None` where none runs.
    pub(crate) tree: Option<DraftTree>,
    /// Rows of `tree` computed by earlier forwards.
    pub(crate) read_before: usize,
    /// Per row run: does its draw go back?
    answered: Vec<bool>,
    /// The forward's row count.
    positions: usize,
    /// What to keep once the forward has run.
    next: LiveRows,
}

impl ServeRows {
    /// The cache position the rows run from.
    pub(crate) fn index_pos(&self) -> usize {
        self.next.from + self.read_before
    }

    /// The forward's answer — per row, its draw where it goes back,
    /// [`NOT_DRAWN`] elsewhere — from the rows run's draws (`draws[i]` is
    /// `rows[i]`'s), and what to keep. A row drawn before keeps the draw sent
    /// then ([`SentBefore`]).
    pub(crate) fn answer(mut self, draws: &[u32]) -> Result<(Vec<u32>, LiveRows), String> {
        if draws.len() != self.rows.len() {
            return Err(format!(
                "skipping rows: {} draws for {} rows run",
                draws.len(),
                self.rows.len()
            ));
        }
        let first = self.next.local.len() - self.positions;
        let mut out = vec![NOT_DRAWN; self.positions];
        for ((&row, &draw), &answered) in self.rows.iter().zip(draws).zip(&self.answered) {
            if answered {
                out[row] = *self.next.draws[first + row].get_or_insert(draw);
            }
        }
        Ok((out, self.next))
    }
}

/// One change to a tree that flows, in the order the requester made it. The
/// requester's copy of the tree and the drafter's apply the same list in the
/// same order, with the drafter's proposals — rows of both the moment they are
/// made, never an op — between them, so a row number means the same row on
/// both, and on every segment, which reads the rows in that order.
#[derive(serde::Serialize, serde::Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
pub enum FlowOp {
    /// A row carrying `token` after row `parent`: the model's draw no row
    /// carried (`WalkEnd::Missed`).
    Push { parent: u16, token: u32 },
    /// A row carrying `token` after row `parent`, copied from what followed
    /// the reply's last tokens earlier in the prompt or the reply (`dsd_flow`'s
    /// copied runs). A row as a push is, but no surer than the drafter's own
    /// guess: it is rated by that once its parent is read.
    Copy { parent: u16, token: u32 },
    /// The walk now stands on `row`, the rows before it on its path emitted.
    Advance { row: u16 },
    /// Cut the tree to the walk's row and the rows following it
    /// ([`FlowTree::rebase`]).
    Rebase,
}

/// What [`FlowTree::rebase`] did, for a caller keeping something per row.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Rebased {
    /// Where the tree started before.
    pub old_base: usize,
    /// The tokens on the walk's path before its row — emitted, now context:
    /// a segment reads them again from `old_base` after cutting its cache there.
    pub path: Vec<u32>,
    /// The rows kept, as old row numbers, in their new order: the walk's row
    /// (new row 0), then every row following it, as they were laid out.
    pub kept: Vec<usize>,
}

/// A tree of guesses that FLOWS (#134 (b), `docs/plans/large_model_splits.md`
/// § "A tree that flows"): grown and sent a few rows at a time while the
/// requester walks the draws that come back (`walk_draws`). Row 0 is the token
/// the walk started from, at `base`; `at` is the row it stands on now, whose
/// draw has not come back. A row is LIVE when it is `at` or follows it; the
/// rest are lost branches, still in every segment's cache and read by nothing.
///
/// Every segment holds the rows in this order (a forward carries the tree's
/// last rows, `LayerForward::draft_tree_from`), so the only way rows change is
/// [`FlowOp`], applied alike by the requester and the drafter.
#[derive(Debug, Clone)]
pub(crate) struct FlowTree {
    tree: DraftTree,
    tokens: Vec<u32>,
    at: usize,
}

impl FlowTree {
    /// A tree of one row, `root`, at position `base`; the walk on it.
    pub(crate) fn new(base: usize, root: u32) -> Result<Self, String> {
        Ok(Self {
            tree: DraftTree::new(base, vec![None])?,
            tokens: vec![root],
            at: 0,
        })
    }

    pub(crate) fn tree(&self) -> &DraftTree {
        &self.tree
    }

    pub(crate) fn tokens(&self) -> &[u32] {
        &self.tokens
    }

    pub(crate) fn len(&self) -> usize {
        self.tokens.len()
    }

    pub(crate) fn base(&self) -> usize {
        self.tree.base()
    }

    /// The row the walk stands on.
    pub(crate) fn at(&self) -> usize {
        self.at
    }

    /// Is `row` the walk's row or one following it?
    pub(crate) fn is_live(&self, row: usize) -> bool {
        if row >= self.len() {
            return false;
        }
        let floor = self.tree.depth(self.at);
        let mut r = row;
        while self.tree.depth(r) > floor {
            match self.tree.parent(r) {
                Some(p) => r = p,
                None => return false,
            }
        }
        r == self.at
    }

    /// How many rows are live.
    pub(crate) fn live_rows(&self) -> usize {
        (self.at..self.len()).filter(|&r| self.is_live(r)).count()
    }

    /// Does a row following `parent` carry `token` already?
    pub(crate) fn has_child(&self, parent: usize, token: u32) -> bool {
        self.child_carrying(parent, token).is_some()
    }

    /// The row following `parent` that carries `token`, if one does.
    pub(crate) fn child_carrying(&self, parent: usize, token: u32) -> Option<usize> {
        self.tree
            .children(Some(parent))
            .find(|&c| self.tokens[c] == token)
    }

    /// Add a row carrying `token` after `parent`. Refused where a row
    /// following `parent` already carries it — the rows following one row
    /// carry distinct tokens, which the walk's exactness rests on — and where
    /// the tree is full ([`MAX_TREE_ROWS`]): the requester rebases first.
    pub(crate) fn push(&mut self, parent: usize, token: u32) -> Result<usize, String> {
        if parent >= self.len() {
            return Err(format!(
                "a row after row {parent} of a tree of {} rows",
                self.len()
            ));
        }
        if self.has_child(parent, token) {
            return Err(format!("row {parent} is followed by {token} already"));
        }
        let row = self.tree.push(Some(parent))?;
        self.tokens.push(token);
        Ok(row)
    }

    /// The walk now stands on `row`, which must be live: a walk only goes
    /// down.
    pub(crate) fn advance(&mut self, row: usize) -> Result<(), String> {
        if !self.is_live(row) {
            return Err(format!(
                "the walk cannot move from row {} to row {row}",
                self.at
            ));
        }
        self.at = row;
        Ok(())
    }

    /// Cut the tree to its live rows: the walk's row becomes row 0, at the
    /// position it already holds (`base + depth`), and the rows following it
    /// keep their layout. The path before it is emitted and becomes context.
    /// A segment does the same by cutting its cache to `old_base` and reading
    /// `path` then the kept rows again — one forward, a tree from `old_base`
    /// whose first rows are a chain ([`Self::rebase_forward`]).
    pub(crate) fn rebase(&mut self) -> Result<Rebased, String> {
        let old_base = self.base();
        let mut path: Vec<u32> =
            std::iter::successors(self.tree.parent(self.at), |&r| self.tree.parent(r))
                .map(|r| self.tokens[r])
                .collect();
        path.reverse();
        let kept: Vec<usize> = (self.at..self.len()).filter(|&r| self.is_live(r)).collect();
        let mut new_index = vec![usize::MAX; self.len()];
        for (new, &old) in kept.iter().enumerate() {
            new_index[old] = new;
        }
        let parents = kept
            .iter()
            .map(|&old| {
                if old == self.at {
                    None
                } else {
                    self.tree.parent(old).map(|p| new_index[p] as u16)
                }
            })
            .collect();
        let tokens = kept.iter().map(|&old| self.tokens[old]).collect();
        self.tree = DraftTree::new(old_base + path.len(), parents)?;
        self.tokens = tokens;
        self.at = 0;
        Ok(Rebased {
            old_base,
            path,
            kept,
        })
    }

    /// The forward that reads a rebased tree into a segment whose cache was
    /// cut to `rebased.old_base`: the path as a chain, then every row of this
    /// (rebased) tree after the path's last token. Returns its tokens and its
    /// shape as a tree from `old_base` (wire form).
    pub(crate) fn rebase_forward(&self, rebased: &Rebased) -> (Vec<u32>, Vec<u16>) {
        let lead = rebased.path.len();
        let mut tokens = rebased.path.clone();
        tokens.extend(&self.tokens);
        // Row i of the chain follows row i - 1 (wire i); the tree's rows follow
        // the chain's last (wire `lead`) or their own parent, `lead` rows on.
        let mut wire: Vec<u16> = (0..lead as u16).collect();
        wire.extend(self.tree.to_wire().iter().map(|&w| {
            if w == 0 {
                lead as u16
            } else {
                w + lead as u16
            }
        }));
        (tokens, wire)
    }

    /// Apply one change. A rebase answers what it did.
    #[cfg(test)]
    pub(crate) fn apply(&mut self, op: FlowOp) -> Result<Option<Rebased>, String> {
        match op {
            FlowOp::Push { parent, token } | FlowOp::Copy { parent, token } => {
                self.push(usize::from(parent), token).map(|_| None)
            }
            FlowOp::Advance { row } => self.advance(usize::from(row)).map(|()| None),
            FlowOp::Rebase => self.rebase().map(Some),
        }
    }
}

/// Lay a tree out with its most likely path first: rows `0..d` become the
/// chain the drafter rated highest at each step, so the path a check keeps is
/// usually a prefix of the rows and the next forward cuts the cache to it
/// instead of reading those rows again (`docs/plans/large_model_splits.md` §
/// "Draft trees"). Every other row follows in its original order, which keeps
/// each parent before its children. `scores[i]` is row i's rating among its
/// siblings — the drafter's log-probability.
///
/// Returns the new order as old row indices.
pub(crate) fn trunk_first(tree: &DraftTree, scores: &[f32]) -> Vec<usize> {
    let mut trunk = Vec::new();
    let mut at: Option<usize> = None;
    loop {
        let best = tree.children(at).max_by(|&a, &b| {
            scores[a]
                .partial_cmp(&scores[b])
                .unwrap_or(std::cmp::Ordering::Equal)
                // The earlier row on a tie: it was chosen first.
                .then(b.cmp(&a))
        });
        match best {
            Some(row) => {
                trunk.push(row);
                at = Some(row);
            }
            None => break,
        }
    }
    let mut on_trunk = vec![false; tree.len()];
    for &r in &trunk {
        on_trunk[r] = true;
    }
    trunk.extend((0..tree.len()).filter(|&r| !on_trunk[r]));
    trunk
}

/// `tree` and its per-row values with the rows in `order` (old indices, as
/// [`trunk_first`] returns them); parents renumbered to match.
pub(crate) fn reorder<T: Copy>(
    tree: &DraftTree,
    values: &[T],
    order: &[usize],
) -> Result<(DraftTree, Vec<T>), String> {
    let mut new_index = vec![usize::MAX; tree.len()];
    for (new, &old) in order.iter().enumerate() {
        new_index[old] = new;
    }
    let parents = order
        .iter()
        .map(|&old| tree.parent(old).map(|p| new_index[p] as u16))
        .collect();
    let values = order.iter().map(|&old| values[old]).collect();
    Ok((DraftTree::new(tree.base, parents)?, values))
}

/// The words of each guess's marginals a block drafter's tree is grown from
/// (#437, [`GrownTree::from_marginals`]): what a card's ranking kernel ranks a
/// row (`tree_rank::MOST_FOLLOWERS`); a tree of 31 spends few past them.
pub(crate) const MARGINAL_WORDS: usize = 8;

/// A tree of guesses as a drafter grew it (`SplitModel::draft_tree_after`),
/// its likeliest path first: each guess's token, its parent among the guesses
/// (`None`: it follows the last token read) and its path's log-probability
/// under the drafter. A path's rating never rises along it, so the guesses
/// rated above any bar form a tree too ([`worth_checking`]).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct GrownTree {
    pub tokens: Vec<u32>,
    pub parents: Vec<Option<u16>>,
    pub path_logp: Vec<f32>,
}

impl GrownTree {
    /// A chain of guesses as a tree: each follows the one before, the first
    /// the last token read; with no ratings, each is rated sure (path
    /// log-probability 0), so a price that keeps any keeps them in order.
    /// What a check that must read kept tokens again carries guesses that
    /// came as a chain in (#437, `TreeRounds::check`).
    pub(crate) fn chain(tokens: &[u32]) -> Self {
        Self {
            tokens: tokens.to_vec(),
            parents: (0..tokens.len())
                .map(|i| i.checked_sub(1).map(|p| p as u16))
                .collect(),
            path_logp: vec![0.0; tokens.len()],
        }
    }

    /// A tree from a BLOCK drafter's one forward (#437): `rows[d]` is the
    /// drafter's best words at guess position `d` with their log-probability
    /// there, best first. The block's own picks — `rows[d][0]` for every `d`
    /// below `chain` — are taken whole, as the chain it guessed; the rest of
    /// `budget` nodes best-first, a node rated by its path's summed
    /// log-probabilities and its children the next position's words in rank
    /// order (DDTree, arXiv 2604.12989: the block's marginals alone, no
    /// forward per node). Keeping the chain whole is SpecExec's and
    /// Sequoia's: on the 11-18 measure (FUTURE_WORK #437) a pure best-first
    /// tree of 16 kept less than the chain on code (8.31 against 8.93 tokens
    /// a check), the chain plus the rest read as well as either. A child is
    /// never rated above its parent, so every node's parent is taken first.
    /// Laid out likeliest path first (`trunk_first`), so the chain is the
    /// path a check holds in place (`TreeOutcome::rows_in_place`).
    pub(crate) fn from_marginals(rows: &[Vec<(u32, f32)>], chain: usize, budget: usize) -> Self {
        /// A frontier entry's rating, ordered for the max-heap.
        #[derive(PartialEq)]
        struct Rated(f32);
        impl Eq for Rated {}
        impl PartialOrd for Rated {
            fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
                Some(self.cmp(other))
            }
        }
        impl Ord for Rated {
            fn cmp(&self, other: &Self) -> std::cmp::Ordering {
                self.0.total_cmp(&other.0)
            }
        }
        let depths = rows.iter().take_while(|r| !r.is_empty()).count();
        let budget = budget.min(MAX_TREE_ROWS);
        // (parent node, depth, rank in its row, path rating).
        let mut nodes: Vec<(Option<usize>, usize, usize, f32)> = Vec::new();
        // (rating, then the EARLIER entry on a tie, parent, depth, rank).
        type Offer = (Rated, std::cmp::Reverse<usize>, Option<usize>, usize, usize);
        let mut frontier: std::collections::BinaryHeap<Offer> = std::collections::BinaryHeap::new();
        let mut pushed = 0usize;
        let mut offer = |frontier: &mut std::collections::BinaryHeap<_>,
                         rating: f32,
                         parent: Option<usize>,
                         depth: usize,
                         rank: usize| {
            if depth < depths && rank < rows[depth].len() && rating.is_finite() {
                frontier.push((
                    Rated(rating),
                    std::cmp::Reverse(pushed),
                    parent,
                    depth,
                    rank,
                ));
                pushed += 1;
            }
        };
        let mut rating = 0.0f32;
        for (depth, row) in rows.iter().take(chain.min(depths).min(budget)).enumerate() {
            if !row[0].1.is_finite() {
                break;
            }
            let parent = depth.checked_sub(1);
            offer(
                &mut frontier,
                rating + row.get(1).map_or(f32::NEG_INFINITY, |w| w.1),
                parent,
                depth,
                1,
            );
            rating += row[0].1;
            nodes.push((parent, depth, 0, rating));
        }
        match nodes.last() {
            Some(&(_, depth, _, rating)) => offer(
                &mut frontier,
                rating
                    + rows
                        .get(depth + 1)
                        .and_then(|r| r.first())
                        .map_or(f32::NEG_INFINITY, |w| w.1),
                Some(nodes.len() - 1),
                depth + 1,
                0,
            ),
            None => offer(
                &mut frontier,
                rows.first()
                    .and_then(|r| r.first())
                    .map_or(f32::NEG_INFINITY, |w| w.1),
                None,
                0,
                0,
            ),
        }
        while nodes.len() < budget {
            let Some((Rated(rating), _, parent, depth, rank)) = frontier.pop() else {
                break;
            };
            let at = nodes.len();
            nodes.push((parent, depth, rank, rating));
            let parent_rating = rating - rows[depth][rank].1;
            if let Some(next) = rows[depth].get(rank + 1) {
                offer(
                    &mut frontier,
                    parent_rating + next.1,
                    parent,
                    depth,
                    rank + 1,
                );
            }
            if let Some(first) = rows.get(depth + 1).and_then(|r| r.first()) {
                offer(&mut frontier, rating + first.1, Some(at), depth + 1, 0);
            }
        }
        let parents: Vec<Option<u16>> = nodes.iter().map(|n| n.0.map(|p| p as u16)).collect();
        let Ok(tree) = DraftTree::new(1, parents) else {
            return Self::default();
        };
        let own: Vec<f32> = nodes.iter().map(|n| rows[n.1][n.2].1).collect();
        let order = trunk_first(&tree, &own);
        let values: Vec<(u32, f32)> = nodes.iter().map(|n| (rows[n.1][n.2].0, n.3)).collect();
        let Ok((laid, values)) = reorder(&tree, &values, &order) else {
            return Self::default();
        };
        Self {
            parents: (0..laid.len())
                .map(|r| laid.parent(r).map(|p| p as u16))
                .collect(),
            tokens: values.iter().map(|&(t, _)| t).collect(),
            path_logp: values.iter().map(|&(_, s)| s).collect(),
        }
    }

    /// The part of this tree that follows `kept` and then `next` — the guesses
    /// a check that kept `kept` and sampled `next` can send at once, with no
    /// drafting in its way. `None` where the tree holds no such path.
    ///
    /// While a check is on its way, the drafter keeps growing the tree it was
    /// sent from (`dsd_tree`): best-first by path probability, which is each
    /// guess's chance of lying on the reply's real path — in this check if it
    /// was sent, else in the next one, as part of the subtree under the token
    /// the target samples where its walk stops. That subtree is the next
    /// check's tree. Speculative Speculative Decoding (arXiv 2603.03251) caches
    /// the next round per outcome for chains; a tree grown on is that cache.
    ///
    /// Ratings are made relative to the new root, at most `most` kept (the
    /// best; still a tree), likeliest path first (`trunk_first`).
    pub(crate) fn after(&self, kept: &[u32], next: u32, most: usize) -> Option<GrownTree> {
        let n = self.tokens.len();
        if self.parents.len() != n || self.path_logp.len() != n {
            return None;
        }
        let parent_of = |i: usize| self.parents[i].map(usize::from);
        let mut root: Option<usize> = None;
        for &t in kept.iter().chain(std::iter::once(&next)) {
            root = Some((0..n).find(|&i| parent_of(i) == root && self.tokens[i] == t)?);
        }
        let root = root?;
        // Laid out parents first, so one pass finds every row under the root.
        let mut new_index = vec![None; n];
        let mut under: Vec<(usize, Option<u16>)> = Vec::new();
        for i in 0..n {
            let parent = match parent_of(i) {
                Some(p) if p == root => Some(None),
                Some(p) if p < i => new_index[p].map(|q: usize| Some(q as u16)),
                _ => None,
            };
            if let Some(parent) = parent {
                new_index[i] = Some(under.len());
                under.push((i, parent));
            }
        }
        let base = self.path_logp[root];
        let rated: Vec<f32> = under
            .iter()
            .map(|&(i, _)| self.path_logp[i] - base)
            .collect();
        // The `most` best: a child is never rated above its parent and sits
        // after it, so they are a tree too.
        let mut best: Vec<usize> = (0..under.len()).collect();
        best.sort_by(|&a, &b| rated[b].total_cmp(&rated[a]).then(a.cmp(&b)));
        best.truncate(most);
        best.sort_unstable();
        let parents = keep_rows(&under.iter().map(|&(_, p)| p).collect::<Vec<_>>(), &best)?;
        // The base only places rows (a tree follows at least one position);
        // the shape reads the same at any.
        let tree = DraftTree::new(1, parents).ok()?;
        let rated: Vec<f32> = best.iter().map(|&r| rated[r]).collect();
        let own: Vec<f32> = (0..tree.len())
            .map(|r| rated[r] - tree.parent(r).map_or(0.0, |p| rated[p]))
            .collect();
        let order = trunk_first(&tree, &own);
        let values: Vec<(u32, f32)> = best
            .iter()
            .zip(&rated)
            .map(|(&r, &s)| (self.tokens[under[r].0], s))
            .collect();
        let (laid, values) = reorder(&tree, &values, &order).ok()?;
        Some(GrownTree {
            parents: (0..laid.len())
                .map(|r| laid.parent(r).map(|p| p as u16))
                .collect(),
            tokens: values.iter().map(|&(t, _)| t).collect(),
            path_logp: values.iter().map(|&(_, s)| s).collect(),
        })
    }
}

/// What a check costs, as the round loop measured it (`CheckCost::fit`): a
/// fixed part — the round trip and every segment's per-forward work — and a
/// part per row it carries (each row's hidden state crosses every link).
#[derive(Debug, Clone, Copy)]
pub(crate) struct CheckPrice {
    pub fixed_ms: f64,
    pub per_row_ms: f64,
}

/// The guesses of a grown tree worth a place in the check, as indices in
/// their original order (closed under parents, so still a tree).
///
/// A guess is expected to be kept with its path's probability — SpecExec's
/// and EAGLE-2's reading of a drafter's confidence (arXiv 2406.02532,
/// 2406.16858) — scaled by `calibration`, how far this drafter's ratings have
/// matched what checks kept. The check yields those expected guesses plus the
/// token it samples, and costs `fixed + (lead_rows + n) × per_row`; the
/// guesses kept are the `n` most likely that maximise tokens per millisecond.
/// `lead_rows` are the rows the check carries anyway (kept tokens read again,
/// its own last token). Ratings sort a parent before its children (a child is
/// never rated above it; on a tie the parent's index is lower), so any `n`
/// most likely are a tree. At most `most` are kept.
pub(crate) fn worth_checking(
    path_logp: &[f32],
    lead_rows: usize,
    price: CheckPrice,
    calibration: f64,
    most: usize,
) -> Vec<usize> {
    let mut by_rating: Vec<usize> = (0..path_logp.len()).collect();
    by_rating.sort_by(|&a, &b| path_logp[b].total_cmp(&path_logp[a]).then(a.cmp(&b)));
    let mut expected = 1.0;
    let mut best = (
        expected / (price.fixed_ms + lead_rows as f64 * price.per_row_ms),
        0,
    );
    for (n, &row) in by_rating.iter().take(most).enumerate() {
        expected += (calibration * f64::from(path_logp[row]).exp()).clamp(0.0, 1.0);
        let cost = price.fixed_ms + (lead_rows + n + 1) as f64 * price.per_row_ms;
        let rate = expected / cost.max(f64::MIN_POSITIVE);
        if rate > best.0 {
            best = (rate, n + 1);
        }
    }
    let mut kept = by_rating[..best.1].to_vec();
    kept.sort_unstable();
    kept
}

/// The parents of the rows `kept` (ascending) of a tree, renumbered among
/// them; `None` when a kept row's parent was not kept — no longer a tree.
pub(crate) fn keep_rows(parents: &[Option<u16>], kept: &[usize]) -> Option<Vec<Option<u16>>> {
    let mut new_index = vec![None; parents.len()];
    for (new, &old) in kept.iter().enumerate() {
        *new_index.get_mut(old)? = Some(new as u16);
    }
    kept.iter()
        .map(|&old| match parents[old] {
            None => Some(None),
            Some(p) => new_index.get(usize::from(p)).copied().flatten().map(Some),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Row 0 follows the context; 1 and 2 follow it (tokens 10, 11); 3 follows
    /// 1, 4 follows 2, 5 follows 3, 6 follows 4.
    fn two_branches() -> (DraftTree, Vec<u32>) {
        let parents = vec![None, Some(0), Some(0), Some(1), Some(2), Some(3), Some(4)];
        let tree = DraftTree::new(100, parents).unwrap();
        (tree, vec![1, 10, 11, 20, 21, 30, 31])
    }

    #[test]
    fn a_far_computes_only_the_rows_its_own_draws_leave_on_the_walk() {
        let (tree, tokens) = two_branches();
        let mut draws = vec![None; 7];
        // No draw back yet: from row 0 every later row may be reached.
        assert_eq!(
            rows_the_walk_may_reach(&tree, &tokens, &draws, 0, 3..7),
            vec![3, 4, 5, 6]
        );
        // Row 0 drew 10: the walk takes row 1, and the branch under 2 is dead.
        draws[0] = Some(10);
        assert_eq!(
            rows_the_walk_may_reach(&tree, &tokens, &draws, 0, 3..7),
            vec![3, 5]
        );
        // The requester had already walked to row 2 when it sent the forward.
        assert_eq!(
            rows_the_walk_may_reach(&tree, &tokens, &[None; 7], 2, 3..7),
            vec![4, 6]
        );
        // A draw no row carries ends the walk: nothing is reached.
        draws[0] = Some(99);
        assert!(rows_the_walk_may_reach(&tree, &tokens, &draws, 0, 3..7).is_empty());
    }

    #[test]
    fn a_far_forward_runs_its_live_rows_padded_to_one_shape() {
        // Two of 24 live: padded to 8 with the forward's first other rows.
        assert_eq!(
            rows_to_serve(&[3, 5], 0..24, 8),
            vec![0, 1, 2, 3, 4, 5, 6, 7]
        );
        assert_eq!(
            rows_to_serve(&[17, 20], 16..40, 8),
            vec![16, 17, 18, 19, 20, 21, 22, 23]
        );
        // None live: none run.
        assert!(rows_to_serve(&[], 0..24, 8).is_empty());
        // A forward no wider than the shape runs whole.
        assert_eq!(rows_to_serve(&[2], 0..6, 8), vec![0, 1, 2, 3, 4, 5]);
        // More live than the shape holds: exactly those.
        let live: Vec<usize> = (0..24).step_by(2).collect();
        assert_eq!(rows_to_serve(&live, 0..24, 8), live);
    }

    /// `two_branches` grown by two rows: 7 follows 6, 8 follows 5.
    fn two_branches_grown() -> (DraftTree, Vec<u32>) {
        let parents = vec![
            None,
            Some(0),
            Some(0),
            Some(1),
            Some(2),
            Some(3),
            Some(4),
            Some(6),
            Some(5),
        ];
        let tree = DraftTree::new(100, parents).unwrap();
        (tree, vec![1, 10, 11, 20, 21, 30, 31, 40, 41])
    }

    #[test]
    fn a_far_keeps_only_the_live_rows_and_answers_only_sound_ones() {
        let (tree, tokens) = two_branches();
        let three = DraftTree::new(100, vec![None, Some(0), Some(0)]).unwrap();
        // First forward, rows 0-2: nothing drawn yet, all run whole.
        let first = LiveRows::plan(None, &three, &tokens[..3], 0, 3, 8).unwrap();
        assert_eq!(first.rows, vec![0, 1, 2]);
        assert_eq!(first.index_pos(), 100);
        // Row 0 drew 10 (row 1's token), row 1 drew 20 (row 3's), row 2 99.
        let (out, kept) = first.answer(&[10, 20, 99]).unwrap();
        assert_eq!(out, vec![10, 20, 99]);
        // Second forward, rows 3-6, sent while the requester stood on row 0:
        // the walk here is at row 3, so 3 and 5 are live, 4 and 6 dead.
        let second = LiveRows::plan(Some(kept.clone()), &tree, &tokens, 0, 4, 2).unwrap();
        assert_eq!(second.rows, vec![0, 2]);
        assert_eq!(second.read_before, 3);
        assert_eq!(second.index_pos(), 103);
        let compact = second.tree.clone().unwrap();
        assert_eq!(compact.len(), 5);
        assert_eq!(compact.parent(3), Some(1));
        assert_eq!(compact.parent(4), Some(3));
        assert_eq!(compact.depth(4), tree.depth(5));
        let (out, kept2) = second.answer(&[30, 7]).unwrap();
        assert_eq!(out, vec![30, NOT_DRAWN, 7, NOT_DRAWN]);
        assert_eq!(kept2.local_position(107), 105);
        assert_eq!(kept2.local_position(104), 104);
        assert_eq!(kept2.local_position(99), 99);
        // Padded to one shape: row 4 runs beside the live ones, its draw kept back.
        let padded = LiveRows::plan(Some(kept.clone()), &tree, &tokens, 0, 4, 3).unwrap();
        assert_eq!(padded.rows, vec![0, 1, 2]);
        let (out, _) = padded.answer(&[30, 5, 7]).unwrap();
        assert_eq!(out, vec![30, NOT_DRAWN, 7, NOT_DRAWN]);
        // Third forward, rows 7-8, from a requester whose walk stands on the
        // skipped row 6: row 7 follows it but its path lost rows 4 and 6, so
        // its draw is never sent; row 8 is off that walk.
        let (grown, grown_tokens) = two_branches_grown();
        let third = LiveRows::plan(Some(kept2.clone()), &grown, &grown_tokens, 6, 2, 8).unwrap();
        assert_eq!(third.rows, vec![0, 1]);
        let compact = third.tree.clone().unwrap();
        assert_eq!(
            compact.parent(5),
            Some(2),
            "row 7 hangs off the nearest row computed"
        );
        let (out, _) = third.answer(&[1, 2]).unwrap();
        assert_eq!(out, vec![NOT_DRAWN, NOT_DRAWN]);
        // A walk that missed: nothing in this forward is live, nothing runs.
        let mut missed = kept.clone();
        missed.draws[0] = Some(77);
        let none = LiveRows::plan(Some(missed), &tree, &tokens, 0, 4, 8).unwrap();
        assert!(none.rows.is_empty() && none.tree.is_none());
        let (out, after) = none.answer(&[]).unwrap();
        assert_eq!(out, vec![NOT_DRAWN; 4]);
        assert_eq!(after.local_position(107), 103);
    }

    #[test]
    fn a_rebase_forward_runs_whole_and_answers_its_kept_rows() {
        // A rebase's forward: two rows of the path the walk took (0, 1), then
        // the kept rows from the walk's (2) on — 3 and 4 follow it, 5 follows 3.
        let tree =
            DraftTree::new(300, vec![None, Some(0), Some(1), Some(2), Some(2), Some(3)]).unwrap();
        let tokens = vec![5, 6, 7, 8, 9, 10];
        let plan = LiveRows::plan(None, &tree, &tokens, 2, 6, 1).unwrap();
        // Whole, though only the walk's rows are live and the shape holds one.
        assert_eq!(plan.rows, vec![0, 1, 2, 3, 4, 5]);
        assert_eq!(plan.index_pos(), 300);
        let (out, kept) = plan.answer(&[1, 2, 8, 10, 5, 11]).unwrap();
        // The path's draws stay home: the walk stands past them.
        assert_eq!(out, vec![NOT_DRAWN, NOT_DRAWN, 8, 10, 5, 11]);
        assert_eq!(kept.local_position(306), 306);
        // The next forward: the tree now starts where the path ends (302), its
        // first four rows the kept ones (2-5), and a fifth follows row 3 (old
        // 5). The far's draws walk 0 → 1 → 3 → 4: only the new row is live.
        let next = DraftTree::new(302, vec![None, Some(0), Some(0), Some(1), Some(3)]).unwrap();
        let plan = LiveRows::plan(Some(kept), &next, &[7, 8, 9, 10, 11], 0, 1, 1).unwrap();
        // The forward's one row (the tree's row 4), run past the four kept.
        assert_eq!(plan.rows, vec![0]);
        assert_eq!(plan.index_pos(), 306);
    }

    /// A rebase's forward reads its kept rows again in one forward of another
    /// row count; on a card a near-tie can draw the other way. The requester
    /// keeps the first draw, so the far must answer and walk by it too, or the
    /// two walks part and the flow ends (2026-11-28, TH→AU: 2 replies of 5).
    #[test]
    fn a_row_read_again_after_a_rebase_keeps_the_draw_sent_first() {
        // Root 5 at 100; 9 and 11 follow it.
        let three = DraftTree::new(100, vec![None, Some(0), Some(0)]).unwrap();
        let first = LiveRows::plan(None, &three, &[5, 9, 11], 0, 3, 8).unwrap();
        let (out, kept) = first.answer(&[9, 13, 99]).unwrap();
        assert_eq!(out, vec![9, 13, 99]);
        // 13 and 17 follow 9, 21 follows 13: the walk here is at row 3 (13).
        let tree =
            DraftTree::new(100, vec![None, Some(0), Some(0), Some(1), Some(1), Some(3)]).unwrap();
        let tokens = [5, 9, 11, 13, 17, 21];
        let second = LiveRows::plan(Some(kept), &tree, &tokens, 0, 3, 8).unwrap();
        let (out, kept) = second.answer(&[21, 0, 25]).unwrap();
        assert_eq!(out, vec![21, NOT_DRAWN, 25]);
        // The requester walked 5 → 9 → 13 and rebases there: its cache and
        // this one are cut to 100, then read again — the path (5, 9), the kept
        // rows (13, 21) and one new row (40 after 21).
        let cut = kept.truncated_to(100).unwrap();
        let again = DraftTree::new(100, vec![None, Some(0), Some(1), Some(2), Some(3)]).unwrap();
        let rebase = LiveRows::plan(Some(cut), &again, &[5, 9, 13, 21, 40], 2, 5, 8).unwrap();
        assert_eq!(
            rebase.rows,
            vec![0, 1, 2, 3, 4],
            "a rebase's forward runs whole"
        );
        // Drawn again, 13 and 21 come out the other way (14, 26): the draws
        // sent first stand; the new row's is its own.
        let (out, kept) = rebase.answer(&[1, 2, 14, 26, 50]).unwrap();
        assert_eq!(out, vec![NOT_DRAWN, NOT_DRAWN, 21, 25, 50]);
        // The next forward, sent from the kept rows' first (13): 25 follows 21
        // (the walk's), 77 follows 13. Walked by the first draws, 13 → 21 → 25:
        // row 3 is live and drawn; walked by 14 it would have missed at 13 and
        // left row 3 undrawn where the requester's walk reaches it.
        let next = DraftTree::new(102, vec![None, Some(0), Some(1), Some(1), Some(0)]).unwrap();
        let plan = LiveRows::plan(Some(kept), &next, &[13, 21, 40, 25, 77], 0, 2, 8).unwrap();
        let (out, _) = plan.answer(&[60, 61]).unwrap();
        assert_eq!(out, vec![60, NOT_DRAWN]);
    }

    #[test]
    fn rows_kept_start_a_later_tree_only_where_none_before_it_was_skipped() {
        let held = LiveRows {
            from: 300,
            local: vec![Some(0), None, Some(1)],
            sound: vec![true, false, true],
            draws: vec![Some(4), None, Some(6)],
            ..LiveRows::default()
        };
        // Row 1 was skipped: the cache does not hold position 301 where a tree
        // from 302 would say.
        assert!(held.clone().rebased_to(302).is_none());
        assert!(held.clone().rebased_to(299).is_none(), "before the tree");
        assert!(held.clone().rebased_to(304).is_none(), "past what is held");
        // From 301 the rows keep their slots: a position reads alike either way.
        let later = held.clone().rebased_to(301).unwrap();
        assert_eq!(later.local, vec![None, Some(0)]);
        assert_eq!(later.draws, vec![None, Some(6)]);
        assert_eq!(later.local_position(303), held.local_position(303));
        assert_eq!(held.clone().rebased_to(300), Some(held));
        // A map whose slots break the count (none this far builds) is refused,
        // never wrapped round (security review, 2026-11-28).
        let broken = LiveRows {
            from: 300,
            local: vec![Some(0), Some(0)],
            sound: vec![true, true],
            draws: vec![None, None],
            ..LiveRows::default()
        };
        assert!(broken.rebased_to(301).is_none());
    }

    #[test]
    fn a_far_refuses_rows_it_kept_a_different_count_of() {
        let (tree, tokens) = two_branches();
        let three = DraftTree::new(100, vec![None, Some(0), Some(0)]).unwrap();
        let (_, kept) = LiveRows::plan(None, &three, &tokens[..3], 0, 3, 8)
            .unwrap()
            .answer(&[10, 20, 99])
            .unwrap();
        // Five rows read before, three kept.
        assert!(LiveRows::plan(Some(kept.clone()), &tree, &tokens, 0, 2, 8).is_err());
        // Rows kept of another tree are not this one's: refused.
        let other = DraftTree::new(
            200,
            vec![None, Some(0), Some(0), Some(1), Some(2), Some(3), Some(4)],
        )
        .unwrap();
        assert!(LiveRows::plan(Some(kept.clone()), &other, &tokens, 0, 4, 8).is_err());
        // Nothing kept: the earlier rows ran whole, every one is in the cache.
        let whole = LiveRows::plan(None, &other, &tokens, 0, 4, 8).unwrap();
        assert_eq!(whole.read_before, 3);
        assert_eq!(whole.index_pos(), 203);
        // Taken back into the tree, and to before it.
        let cut = kept.clone().truncated_to(102).unwrap();
        assert_eq!(cut.local_position(102), 102);
        assert!(kept.truncated_to(99).is_none());
    }

    /// A block drafter's best words at three positions, best first.
    fn marginals() -> Vec<Vec<(u32, f32)>> {
        vec![
            vec![(10, -0.1), (11, -1.0), (12, -3.0)],
            vec![(20, -0.5), (21, -0.7), (22, -4.0)],
            vec![(30, -2.0), (31, -2.1)],
        ]
    }

    /// The block's own picks are taken whole and laid out first, in place;
    /// the rest of the budget goes best-first by path rating — worked by
    /// hand: after the chain 10 → 20 → 30 the frontier rates 21 under 10 at
    /// −0.8, 11 at the root at −1.0, then 20 under 11 at −1.5.
    #[test]
    fn a_block_drafters_tree_keeps_its_chain_and_spends_the_rest_best_first() {
        let rows = marginals();
        let chain_only = GrownTree::from_marginals(&rows, 3, 3);
        assert_eq!(chain_only.tokens, vec![10, 20, 30]);
        assert_eq!(chain_only.parents, vec![None, Some(0), Some(1)]);

        let tree = GrownTree::from_marginals(&rows, 3, 5);
        assert_eq!(tree.tokens, vec![10, 20, 30, 21, 11]);
        assert_eq!(tree.parents, vec![None, Some(0), Some(1), Some(0), None]);
        let want = [-0.1f32, -0.6, -2.6, -0.8, -1.0];
        for (got, want) in tree.path_logp.iter().zip(want) {
            assert!((got - want).abs() < 1e-6, "{:?}", tree.path_logp);
        }

        let tree = GrownTree::from_marginals(&rows, 3, 6);
        assert_eq!(tree.tokens, vec![10, 20, 30, 21, 11, 20]);
        assert_eq!(
            tree.parents,
            vec![None, Some(0), Some(1), Some(0), None, Some(4)]
        );

        // With no chain kept the tree is DDTree's alone: 10, 20, then 21
        // (−0.8) before 11 (−1.0).
        let tree = GrownTree::from_marginals(&rows, 0, 3);
        assert_eq!(tree.tokens, vec![10, 20, 21]);
        assert_eq!(tree.parents, vec![None, Some(0), Some(0)]);

        // Every tree is one a check can carry: parents first, ratings that
        // never rise down a path, siblings with distinct words.
        for budget in 1..12 {
            let t = GrownTree::from_marginals(&rows, 2, budget);
            assert!(t.tokens.len() <= budget);
            assert!(DraftTree::new(1, t.parents.clone()).is_ok());
            for (i, p) in t.parents.iter().enumerate() {
                if let Some(p) = p.map(usize::from) {
                    assert!(t.path_logp[i] <= t.path_logp[p]);
                }
                for j in 0..i {
                    assert!(t.parents[j] != *p || t.tokens[j] != t.tokens[i], "{t:?}");
                }
            }
        }
    }

    /// No words, or a pick the drafter rates as impossible, is no tree.
    /// A chain as a tree (#437): each guess follows the one before, every one
    /// rated sure, so a check priced at any cost keeps them in order.
    #[test]
    fn a_chain_of_guesses_is_a_tree_of_one_path() {
        let chain = GrownTree::chain(&[7, 8, 9]);
        assert_eq!(chain.parents, vec![None, Some(0), Some(1)]);
        assert_eq!(chain.path_logp, vec![0.0; 3]);
        assert_eq!(GrownTree::chain(&[]), GrownTree::default());
    }

    #[test]
    fn a_block_with_no_finite_pick_grows_no_tree() {
        assert!(GrownTree::from_marginals(&[], 4, 8).tokens.is_empty());
        let rows = vec![vec![(1, f32::NEG_INFINITY)], vec![(2, -0.1)]];
        assert!(GrownTree::from_marginals(&rows, 2, 8).tokens.is_empty());
        let rows = vec![vec![(1, -0.1)], vec![(2, f32::NAN)]];
        assert_eq!(GrownTree::from_marginals(&rows, 2, 8).tokens, vec![1]);
    }

    /// Root-level rows 0 and 1; 2 follows 0; 3 follows 2; 4 follows 1.
    fn sample() -> DraftTree {
        DraftTree::new(10, vec![None, None, Some(0), Some(2), Some(1)]).unwrap()
    }

    #[test]
    fn a_row_reads_the_context_and_its_ancestors_only() {
        let t = sample();
        let (positions, seen) = t.rows_for(10, 5).unwrap();
        let mask = seen.mask();
        assert_eq!(positions, vec![10, 10, 11, 12, 11]);
        let kv = 15;
        let reads = |q: usize| -> Vec<usize> {
            (0..kv)
                .filter(|&k| mask[q * kv + k] == 0.0)
                .filter(|&k| k >= 10)
                .map(|k| k - 10)
                .collect()
        };
        assert_eq!(reads(0), vec![0]);
        assert_eq!(reads(1), vec![1]);
        assert_eq!(reads(2), vec![0, 2]);
        assert_eq!(reads(3), vec![0, 2, 3]);
        assert_eq!(reads(4), vec![1, 4]);
        for q in 0..5 {
            assert!(
                (0..10).all(|k| mask[q * kv + k] == 0.0),
                "row {q} reads the context"
            );
        }
    }

    #[test]
    fn a_chain_is_the_causal_mask() {
        let t = DraftTree::new(4, vec![None, Some(0), Some(1)]).unwrap();
        assert!(t.is_chain());
        let (positions, seen) = t.rows_for(4, 3).unwrap();
        let mask = seen.mask();
        assert_eq!(positions, vec![4, 5, 6]);
        for q in 0..3 {
            for k in 0..7 {
                let want = if k <= 4 + q { 0.0 } else { f32::NEG_INFINITY };
                assert_eq!(mask[q * 7 + k], want, "q {q} k {k}");
            }
        }
        assert!(!sample().is_chain());
    }

    /// The drafter's later steps: rows already cached read as keys, only the
    /// newest rows as queries.
    #[test]
    fn a_tree_grown_over_several_forwards_reads_its_cached_rows() {
        let t = sample();
        let (positions, seen) = t.rows_for(13, 2).unwrap();
        let mask = seen.mask();
        assert_eq!(positions, vec![12, 11]);
        let kv = 15;
        let tree_keys = |q: usize| -> Vec<usize> {
            (10..kv)
                .filter(|&k| mask[q * kv + k] == 0.0)
                .map(|k| k - 10)
                .collect()
        };
        assert_eq!(tree_keys(0), vec![0, 2, 3]);
        assert_eq!(tree_keys(1), vec![1, 4]);
        assert!(t.rows_for(12, 2).is_err(), "rows that do not end the tree");
        assert!(
            t.rows_for(9, 6).is_err(),
            "a cache short of the tree's base"
        );
    }

    /// The bits of a row past the first word: a chain of every row a tree may
    /// hold reads, at its last row, every row before it — and its mask is the
    /// causal one, the bits in every word set where they should be.
    #[test]
    fn a_trees_visibility_reaches_its_last_row() {
        let parents = (0..MAX_TREE_ROWS)
            .map(|i| i.checked_sub(1).map(|p| p as u16))
            .collect();
        let t = DraftTree::new(3, parents).unwrap();
        let (_, seen) = t.rows_for(3, MAX_TREE_ROWS).unwrap();
        assert_eq!(seen.rows(), MAX_TREE_ROWS);
        let last = MAX_TREE_ROWS - 1;
        assert!(seen.words[last * VISIBILITY_WORDS..]
            .iter()
            .all(|&w| w == u32::MAX));
        let mask = seen.mask();
        let kv = 3 + MAX_TREE_ROWS;
        for q in [0, 31, 32, 33, 300, last] {
            for k in 0..kv {
                let want = if k <= 3 + q { 0.0 } else { f32::NEG_INFINITY };
                assert_eq!(mask[q * kv + k], want, "q {q} k {k}");
            }
        }
    }

    #[test]
    fn a_parent_after_its_child_or_a_tree_too_large_is_refused() {
        assert!(DraftTree::new(3, vec![None, Some(1)]).is_err());
        assert!(DraftTree::new(3, vec![Some(0)]).is_err());
        assert!(DraftTree::new(0, vec![None]).is_err());
        assert!(DraftTree::new(3, vec![None; MAX_TREE_ROWS + 1]).is_err());
        assert!(DraftTree::from_wire(3, &[0, 5]).is_err());
        let t = sample();
        assert_eq!(DraftTree::from_wire(10, &t.to_wire()).unwrap(), t);
    }

    #[test]
    fn the_kept_tokens_are_read_back_as_rows() {
        // Row 0 is the check's own token; rows 1.. are guesses: 1 and 2 follow
        // it, 3 follows 1, 4 follows 2.
        let t = DraftTree::new(10, vec![None, Some(0), Some(0), Some(1), Some(2)]).unwrap();
        let tokens = [7, 20, 21, 30, 31];
        assert_eq!(kept_rows(&t, 1, &tokens, &[21, 31]), Some(vec![2, 4]));
        assert_eq!(kept_rows(&t, 1, &tokens, &[20]), Some(vec![1]));
        assert_eq!(kept_rows(&t, 1, &tokens, &[]), Some(vec![]));
        assert_eq!(
            kept_rows(&t, 1, &tokens, &[20, 31]),
            None,
            "31 does not follow 20"
        );
        assert_eq!(kept_rows(&t, 1, &tokens, &[7]), None, "no guess carries it");
    }

    #[test]
    fn the_likeliest_path_is_laid_out_first() {
        // 0 and 1 at the root (1 likelier); 2 follows 0; 3 and 4 follow 1 (4 likelier).
        let t = DraftTree::new(5, vec![None, None, Some(0), Some(1), Some(1)]).unwrap();
        let scores = [-1.0, -0.2, -0.1, -2.0, -0.5];
        let order = trunk_first(&t, &scores);
        assert_eq!(order, vec![1, 4, 0, 2, 3]);
        let (laid, tokens) = reorder(&t, &[100, 101, 102, 103, 104], &order).unwrap();
        assert_eq!(tokens, vec![101, 104, 100, 102, 103]);
        assert_eq!(laid.parent(0), None);
        assert_eq!(laid.parent(1), Some(0));
        assert_eq!(laid.parent(2), None);
        assert_eq!(laid.parent(3), Some(2));
        assert_eq!(laid.parent(4), Some(0));
    }

    fn ln(p: &[f64]) -> Vec<f32> {
        p.iter().map(|&p| p.ln() as f32).collect()
    }

    #[test]
    fn a_guess_is_checked_only_while_it_pays_for_its_row() {
        // 0 at the root, 1 and 2 follow 0, 3 follows 1.
        let rated = ln(&[0.9, 0.5, 0.81, 0.1]);
        let price = |per_row_ms| CheckPrice {
            fixed_ms: 400.0,
            per_row_ms,
        };
        // (1 + 0.9 + 0.81 + 0.5) / (400 + 4 × 20) beats adding the 0.1 guess
        // (3.31 / 500) and every smaller tree.
        assert_eq!(
            worth_checking(&rated, 1, price(20.0), 1.0, usize::MAX),
            vec![0, 1, 2]
        );
        // A free row: every guess is worth its place.
        assert_eq!(
            worth_checking(&rated, 1, price(0.0), 1.0, usize::MAX),
            vec![0, 1, 2, 3]
        );
        // A row dearer than ten round trips: none is.
        assert!(worth_checking(&rated, 1, price(4000.0), 1.0, usize::MAX).is_empty());
        // Where rows are dear, a drafter whose ratings have run at twice what
        // checks kept is trusted with fewer of them.
        assert_eq!(
            worth_checking(&rated, 1, price(150.0), 1.0, usize::MAX),
            vec![0, 1, 2]
        );
        assert_eq!(
            worth_checking(&rated, 1, price(150.0), 0.5, usize::MAX),
            vec![0, 2]
        );
        // Held to the two likeliest, which are a tree.
        assert_eq!(worth_checking(&rated, 1, price(0.0), 1.0, 2), vec![0, 2]);
    }

    #[test]
    fn the_guesses_worth_checking_are_a_tree() {
        // A child rated as high as its parent (a certain follower) is never
        // kept without it, whichever comes first in the rows.
        let rated = ln(&[0.6, 0.6, 0.2]);
        let kept = worth_checking(
            &rated,
            1,
            CheckPrice {
                fixed_ms: 100.0,
                per_row_ms: 30.0,
            },
            1.0,
            usize::MAX,
        );
        assert_eq!(kept, vec![0, 1]);
        let parents = [None, Some(0), Some(1)];
        assert_eq!(keep_rows(&parents, &kept), Some(vec![None, Some(0)]));
        assert_eq!(
            keep_rows(&parents, &[0, 2]),
            None,
            "2's parent was left out"
        );
        assert_eq!(
            keep_rows(&[None, Some(0), Some(0), Some(1)], &[0, 2, 3]),
            None
        );
        assert_eq!(
            keep_rows(&[None, Some(0), None, Some(2)], &[0, 2, 3]),
            Some(vec![None, None, Some(1)])
        );
    }

    /// Tokens 10 (A) and 20 (B) follow the last token; 30 (C) and 40 (D)
    /// follow A; 50 (E) follows C; 60 (F) follows D; 70 (G) follows B.
    fn grown_on() -> GrownTree {
        GrownTree {
            tokens: vec![10, 30, 50, 20, 40, 60, 70],
            parents: vec![None, Some(0), Some(1), None, Some(0), Some(4), Some(3)],
            path_logp: vec![-0.1, -0.3, -0.5, -2.0, -1.5, -1.6, -2.2],
        }
    }

    /// The tree a check that kept `kept` and sampled `next` sends at once is
    /// the part grown under that path, rated from its new root, likeliest
    /// path first; a path the tree does not hold has none.
    #[test]
    fn the_tree_grown_on_gives_the_next_check_the_part_under_its_outcome() {
        let g = grown_on();
        // Kept nothing, sampled A: C → E is the trunk, then D → F.
        let next = g.after(&[], 10, usize::MAX).unwrap();
        assert_eq!(next.tokens, vec![30, 50, 40, 60]);
        assert_eq!(next.parents, vec![None, Some(0), None, Some(2)]);
        let want = [-0.2, -0.4, -1.4, -1.5];
        for (got, want) in next.path_logp.iter().zip(want) {
            assert!((got - want).abs() < 1e-6, "{:?}", next.path_logp);
        }
        // Kept A, sampled C: E alone, rated from C.
        let next = g.after(&[10], 30, usize::MAX).unwrap();
        assert_eq!((next.tokens, next.parents), (vec![50], vec![None]));
        // A leaf: nothing grown under it yet.
        assert!(g.after(&[20], 70, usize::MAX).unwrap().tokens.is_empty());
        // A sample the tree never grew, and a kept path it does not hold.
        assert!(g.after(&[10], 99, usize::MAX).is_none());
        assert!(g.after(&[30], 50, usize::MAX).is_none());
        // Held to the best two, which are a tree: C and E.
        let next = g.after(&[], 10, 2).unwrap();
        assert_eq!(
            (next.tokens, next.parents),
            (vec![30, 50], vec![None, Some(0)])
        );
    }
    /// The tokens on `row`'s path in `tree`, root first, itself last.
    fn path_tokens(tree: &DraftTree, tokens: &[u32], row: usize) -> Vec<u32> {
        let mut out: Vec<u32> = std::iter::successors(Some(row), |&r| tree.parent(r))
            .map(|r| tokens[r])
            .collect();
        out.reverse();
        out
    }

    /// A tree that flows (#134 (b)): rows go only down from the walk's row, a
    /// row's followers carry distinct tokens, and a rebase keeps every live
    /// row at the position it held, following the same path — read again in
    /// one forward from the old start, the path as a chain first.
    #[test]
    fn a_tree_that_flows_keeps_its_live_rows_through_a_rebase() {
        // Root 5 at position 40; 9 and 11 follow it; 13 follows 9; 15 follows
        // 11; 17 follows 13; 19 follows 15.
        let mut flow = FlowTree::new(40, 5).unwrap();
        for (parent, token) in [(0, 9), (0, 11), (1, 13), (2, 15), (3, 17), (4, 19)] {
            flow.push(parent, token).unwrap();
        }
        assert!(flow.push(0, 9).is_err(), "row 0 is followed by 9 already");
        assert!(flow.push(9, 1).is_err(), "no row 9");
        assert_eq!(flow.live_rows(), 7);
        // The walk takes 11, then 15: rows 1, 3, 5 are lost.
        flow.advance(2).unwrap();
        flow.advance(4).unwrap();
        assert!(flow.advance(1).is_err(), "a walk only goes down");
        assert!(flow.is_live(6) && flow.is_live(4));
        assert!(!flow.is_live(1) && !flow.is_live(3) && !flow.is_live(5) && !flow.is_live(2));
        assert_eq!(flow.live_rows(), 2);
        // A row the model drew that no row carried: 23 after 15.
        let bonus = flow.push(4, 23).unwrap();
        assert_eq!(bonus, 7);

        let before = flow.clone();
        let rebased = flow.rebase().unwrap();
        assert_eq!(rebased.old_base, 40);
        assert_eq!(rebased.path, vec![5, 11], "the path before the walk's row");
        assert_eq!(rebased.kept, vec![4, 6, 7]);
        assert_eq!(flow.base(), 42);
        assert_eq!(flow.at(), 0);
        assert_eq!(flow.tokens(), &[15, 19, 23]);
        for (new, &old) in rebased.kept.iter().enumerate() {
            assert_eq!(
                flow.base() + flow.tree().depth(new),
                before.base() + before.tree().depth(old),
                "row {old} keeps its position"
            );
        }
        // Read again from the old start: the path as a chain, then the rows.
        let (tokens, wire) = flow.rebase_forward(&rebased);
        let again = DraftTree::from_wire(40, &wire).unwrap();
        assert_eq!(tokens, vec![5, 11, 15, 19, 23]);
        for (new, &old) in rebased.kept.iter().enumerate() {
            let row = rebased.path.len() + new;
            assert_eq!(
                path_tokens(&again, &tokens, row),
                path_tokens(before.tree(), before.tokens(), old),
                "row {old} follows the same path"
            );
            assert_eq!(
                again.base() + again.depth(row),
                40 + before.tree().depth(old)
            );
        }
        // The same changes applied as a list give the same tree.
        let mut replay = FlowTree::new(40, 5).unwrap();
        let ops = [
            FlowOp::Push {
                parent: 0,
                token: 9,
            },
            FlowOp::Push {
                parent: 0,
                token: 11,
            },
            FlowOp::Push {
                parent: 1,
                token: 13,
            },
            FlowOp::Push {
                parent: 2,
                token: 15,
            },
            FlowOp::Push {
                parent: 3,
                token: 17,
            },
            FlowOp::Push {
                parent: 4,
                token: 19,
            },
            FlowOp::Advance { row: 4 },
            FlowOp::Push {
                parent: 4,
                token: 23,
            },
            FlowOp::Rebase,
        ];
        let mut answered = None;
        for op in ops {
            answered = replay.apply(op).unwrap().or(answered);
        }
        assert_eq!(answered, Some(rebased));
        assert_eq!(replay.tokens(), flow.tokens());
        assert_eq!(replay.tree(), flow.tree());
        // A rebase with the walk on row 0 only drops the lost rows.
        let mut root = FlowTree::new(10, 1).unwrap();
        root.push(0, 2).unwrap();
        let r = root.rebase().unwrap();
        assert!(r.path.is_empty());
        assert_eq!((root.base(), root.len()), (10, 2));
        assert_eq!(root.rebase_forward(&r), (vec![1, 2], vec![0, 1]));
    }
}
