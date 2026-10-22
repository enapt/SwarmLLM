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
