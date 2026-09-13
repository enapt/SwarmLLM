'use strict';

// ============================================================================
// SwarmLLM — Dashboard Shard Helpers
// Pure-function HTML builders for shard rows, matrix, coverage ribbon,
// progress bar. No closure state — safe to call from dashboard.js or any
// other component that needs to render shard UI.
//
// Consumes: App.utils (U), App.state (S), I18n.t, I18n.
// Produces: App.dashboardShards.*  — attach-only, no side effects.
// ============================================================================

(function() {
  var S = App.state;
  var U = App.utils;

  var MMPROJ_SHARD_INDEX = 0xFFFFFFFF;
  var MATRIX_MAX_PEERS_DEFAULT = 12;

  /**
   * Build a download progress bar HTML string.
   * @param {Object} opts
   * @param {string} opts.safeId - CSS-safe model ID for data attributes
   * @param {number} opts.pct - Download percentage (0-100)
   * @param {string} opts.label - Left-side label text
   * @param {string} opts.rightText - Right-side text (bytes, speed, etc.)
   * @param {string} [opts.barContent] - Inner bar HTML (segments or fill); defaults to dl-fill
   * @param {string} [opts.cancelBtn] - Optional cancel button HTML appended to right text
   */
  function buildProgressBar(opts) {
    // SEC: escape every text-typed input that lands in innerHTML. Numeric
    // (`pct`) is coerced via `+` and serialized as a number, so the HTML it
    // produces is structurally safe; `barContent` and `cancelBtn` are
    // pre-built HTML strings (callers' responsibility) and are passed
    // through as-is. `safeId`, `label`, and `rightText` are text — escape
    // them. The two current callers only pass server-controlled numerics
    // through `label`/`rightText`, but the function contract should not
    // rely on that — peer-controlled strings (e.g. model display names)
    // shouldn't become an XSS vector simply because someone wires a new
    // caller through.
    var safePct = +opts.pct || 0;
    var safeId = U.escapeHtml(String(opts.safeId || ''));
    var safeLabel = U.escapeHtml(String(opts.label || ''));
    var safeRight = U.escapeHtml(String(opts.rightText || ''));
    var bar = opts.barContent || '<div class="dl-fill" style="width:' + safePct + '%"></div>';
    var right = opts.cancelBtn
      ? '<span style="display:flex;align-items:center;gap:8px"><span class="mono dl-progress-text">' + safeRight + '</span>' + opts.cancelBtn + '</span>'
      : '<span class="mono dl-progress-text">' + safeRight + '</span>';
    return '<div class="dl-progress" data-model-progress="' + safeId + '" data-last-pct="' + safePct + '">' +
      '<div class="flex-between field-hint mb-0">' +
      '<span class="text-muted">' + safeLabel + '</span>' +
      right +
      '</div>' +
      '<div class="dl-bar">' + bar + '</div>' +
      '</div>';
  }

  /**
   * WHOSE MACHINE IS THIS SHARD ON — the single classifier every shard
   * surface reads. One of: live | disk | moving | swarm | thin | absent.
   *
   * This is the question a P2P dashboard exists to answer, and it was being
   * answered in two places that did not agree: the model card graded holders
   * into `thin` (exactly one) vs `swarm` (two or more), while the coverage
   * ribbon did the same with its own copy of the thresholds, and the list and
   * matrix views used a third, coarser vocabulary that could not express
   * "one host away from losing this" at all.
   *
   * Adding a state means adding it here and giving it a `--shard-<state>`
   * colour; do NOT re-derive locality from `holders` at a call site.
   * `shardState` below is the coarse view for the list and matrix, derived
   * from this rather than computed alongside it.
   */
  function shardLocality(s) {
    if (s.local && s.in_vram) return 'live';
    if (s.local) return 'disk';
    if (s.peer_downloads && s.peer_downloads.length > 0) return 'moving';
    var holders = s.holders || 0;
    if (holders >= 2) return 'swarm';
    if (holders === 1) return 'thin';
    return 'absent';
  }

  // Per-shard *display* state for the list and matrix views, which do not
  // distinguish a well-replicated shard from a single-host one. Local
  // download progress is owned by the Downloads panel — never re-rendered
  // inside the model card. The only "in-flight" hint here is gossip from
  // OTHER nodes (peer_downloads), so users can still see the swarm is
  // actively replicating.
  function shardState(s) {
    var loc = shardLocality(s);
    return loc === 'live' ? 'vram'
         : loc === 'disk' ? 'disk'
         : loc === 'moving' ? 'gossip'
         : loc === 'absent' ? 'missing'
         : 'peer';   // swarm + thin both read as "a peer has it"
  }

  // Plain-language tooltip for one piece of the route strip. Says whose
  // machine it is on first, because that is what the colour encodes.
  function shardLocalityLabel(s, loc) {
    if (loc === 'moving') {
      var lead = s.peer_downloads && s.peer_downloads[0] ? (s.peer_downloads[0].progress_pct || 0) : 0;
      return I18n.t('shard.loc.moving') + ' — ' + lead + '%';
    }
    if (loc === 'swarm') return I18n.t('shard.loc.swarm', { n: s.holders || 0 });
    return I18n.t('shard.loc.' + loc);
  }

  /**
   * WHERE ONE PART IS, as the row says it — the single answer, so the builder
   * and the live patcher cannot drift. The wording is the colour key's own
   * `shard.loc.*` string; when the part is on this computer it also says how
   * many OTHER computers have a copy, which is the fact that decides whether
   * losing this machine loses the model. `holders` counts this node, so the
   * "also on" figure subtracts it.
   */
  function shardWhereText(s, loc) {
    var where = shardLocalityLabel(s, loc || shardLocality(s));
    if (!s.local) return where;
    var others = Math.max(0, (s.holders || 0) - 1);
    return where + ' · ' + (others > 0
      ? I18n.t('shard.row.also_on', { count: others })
      : I18n.t('shard.row.only_here'));
  }

  // Torrent-style piece-bar — one colored segment per supplying peer
  function buildPieceBar(peerDownloads, totalPct) {
    if (!peerDownloads || peerDownloads.length === 0) return '';
    var segs = peerDownloads.slice(0, 4);
    var overflow = peerDownloads.length > 4;
    var total = 0;
    segs.forEach(function(p) { total += (p.progress_pct || 0); });
    if (totalPct && total < totalPct) total = totalPct;
    var html = '<div class="shard-row-piecebar">';
    segs.forEach(function(p) {
      var pct = p.progress_pct || 0;
      var color = U.peerColor(p.node_id || '');
      html += '<div class="shard-row-piecebar-seg" style="--w:' + pct + '%;--c:' + color + '" title="' +
        U.escapeHtml((p.node_id || '').substring(0, 12)) + ': ' + pct + '%"></div>';
    });
    if (overflow) {
      html += '<div class="shard-row-piecebar-seg more" style="--w:10%" title="+' + (peerDownloads.length - 4) + '"></div>';
    }
    html += '</div>';
    return html;
  }

  function buildRowActions(state, isLocal, isInVram) {
    var parts = [];
    if (state === 'disk') {
      parts.push('<button class="shard-row-act" data-shard-act="load" title="' + U.escapeHtml(I18n.t('shard.row.action_load')) + '">\u25B2</button>');
    } else if (state === 'vram') {
      parts.push('<button class="shard-row-act" data-shard-act="unload" title="' + U.escapeHtml(I18n.t('shard.row.action_unload')) + '">\u25BC</button>');
    }
    // Local-download state lives in the Downloads panel, not the shard row —
    // no per-row cancel here. Download button is offered for every non-local
    // shard (peer / missing / gossip-in-flight by other peers).
    if (state === 'peer' || state === 'missing' || state === 'gossip') {
      parts.push('<button class="shard-row-act" data-shard-act="download" title="' + U.escapeHtml(I18n.t('shard.download')) + '">\u21E9</button>');
    }
    if (isLocal) {
      parts.push('<button class="shard-row-act danger" data-shard-act="delete" title="' + U.escapeHtml(I18n.t('shard.row.action_delete')) + '">\u2302</button>');
    }
    return parts.length ? '<span class="shard-row-actions">' + parts.join('') + '</span>' : '';
  }

  /**
   * ONE PART OF A MODEL, IN WORDS.
   *
   * The row used to be eight columns of symbols and shouted abbreviations \u2014
   * `\u25aa 1 [1st] LOADED \u25cf+4 509.0 MB \ud83d\udccc \u203a` \u2014 where the only column a reader
   * could decode was the size. Everything it said was true and none of it was
   * legible: "1" did not say of how many, "1st" did not say what being first
   * means, and "\u25cf+4" was a pip whose meaning lived in a hover.
   *
   * It now reads left to right as a sentence: which part, what that part does,
   * where it is, how big. **The "where" is the colour key's own string** \u2014
   * `shard.loc.*`, the same text the legend above the list shows and the same
   * text a strip segment says about itself \u2014 so the three cannot drift apart.
   *
   * The endpoints say what they DO rather than where they sit in an array:
   * the first part turns your prompt into numbers and the last one writes the
   * reply, which is also exactly why holding both is what makes a conversation
   * private. "1st"/"last" stated the position and hid the reason.
   */
  function buildShardRow(s, m, safeId) {
    var state = shardState(s);
    var loc = shardLocality(s);
    var isMmproj = s.index === MMPROJ_SHARD_INDEX;
    var shardCount = m.shard_count || (m.shards || []).length || 0;
    var isFirst = shardCount > 1 && s.index === 0;
    var isLast  = shardCount > 1 && s.index === shardCount - 1;
    var isEndpoint = isFirst || isLast;
    var isPipelinePinned = isEndpoint && s.local && m.encrypted_pipeline;

    var partLabel = isMmproj
      ? I18n.t('shard.row.part_vision')
      : I18n.t('shard.row.part_n_of_m', { n: (s.index || 0) + 1, total: shardCount || 1 });

    var endpointBadge = '';
    if (isFirst) {
      endpointBadge = '<span class="shard-row-endpoint" data-kind="first" title="' + U.escapeHtml(I18n.t('shard.endpoint_first_tip')) + '">' + U.escapeHtml(I18n.t('shard.row.reads_prompt')) + '</span>';
    } else if (isLast) {
      endpointBadge = '<span class="shard-row-endpoint" data-kind="last" title="' + U.escapeHtml(I18n.t('shard.endpoint_last_tip')) + '">' + U.escapeHtml(I18n.t('shard.row.writes_reply')) + '</span>';
    }

    var whereText = shardWhereText(s, loc);
    // The user deleted this piece from this device and has not asked for it
    // since: auto-manage will not bring it back on its own (external report,
    // 2026-08-21 — a deliberate two-machine split was silently undone).
    var removedBadge = (s.removed_by_user && !s.local)
      ? '<span class="shard-row-endpoint" data-kind="removed" title="' + U.escapeHtml(I18n.t('shard.removed_by_user_tip')) + '">' + U.escapeHtml(I18n.t('shard.removed_by_user')) + '</span>'
      : '';
    var sizeText = s.size_bytes ? U.formatBytes(s.size_bytes) : '\u2014';
    var lockCls = s.locked ? ' locked' : '';
    // Pushpin icon = "pin to device" (auto-manage). Reserved 🔒/🔓 for pipeline encryption.
    var lockGlyph = '\uD83D\uDCCC';
    var lockTitle = s.locked ? I18n.t('shard.unlock') : I18n.t('shard.lock');
    var pieceBar = (state === 'gossip' && s.peer_downloads && s.peer_downloads.length > 0)
      ? buildPieceBar(s.peer_downloads, 0)
      : '';
    var actions = buildRowActions(state, !!s.local, !!s.in_vram);
    var rowClass = 'shard-row';
    if (isEndpoint) rowClass += ' shard-row-endpoint-row';
    if (isPipelinePinned) rowClass += ' shard-row-pipeline-pinned';
    return '<div class="' + rowClass + '" data-state="' + state + '"' +
      ' data-shard-row="' + safeId + '-' + s.index + '"' +
      ' data-shard-model="' + U.escapeHtml(m.id) + '"' +
      ' data-shard-index="' + s.index + '"' +
      ' data-shard-locked="' + (s.locked ? '1' : '0') + '">' +
      '<span class="shard-row-part">' + U.escapeHtml(partLabel) + '</span>' +
      '<span class="shard-row-role">' + endpointBadge + removedBadge + '</span>' +
      '<span class="shard-row-where"><span class="avail-seg shard-row-swatch" data-loc="' + loc + '"></span>' +
        U.escapeHtml(whereText) + '</span>' +
      '<span class="shard-row-size">' + sizeText + '</span>' +
      '<button class="shard-row-lock' + lockCls + '" data-shard-act="toggle-lock" title="' + U.escapeHtml(lockTitle) + '">' + lockGlyph + '</button>' +
      '<button class="shard-row-more" data-shard-act="expand" title="' + U.escapeHtml(I18n.t('shard.row.expand_tip')) + '">\u203A</button>' +
      actions +
      pieceBar +
      '</div>';
  }

  function buildShardList(m, shards, safeId) {
    if (!shards || shards.length === 0) return '';
    var rows = shards.map(function(s) { return buildShardRow(s, m, safeId); }).join('');
    return '<div class="shard-list" data-shard-list="' + safeId + '">' + rows + '</div>';
  }

  function buildShardViewToggle() {
    var mode = S._shardView === 'matrix' ? 'matrix' : 'list';
    return '<div class="shard-view-toggle" role="tablist">' +
      '<button type="button" data-shard-view="list" class="' + (mode === 'list' ? 'active' : '') + '" title="' + U.escapeHtml(I18n.t('shard.view.toggle_tip') || '') + '">' + U.escapeHtml(I18n.t('dashboard.info_shards')) + '</button>' +
      '<button type="button" data-shard-view="matrix" class="' + (mode === 'matrix' ? 'active' : '') + '" title="' + U.escapeHtml(I18n.t('shard.view.toggle_tip') || '') + '">' + U.escapeHtml(I18n.t('shard.view.matrix')) + '</button>' +
      '</div>';
  }

  // Matrix view — rows = peers (self pinned top), cols = shards.
  // Cell state derived from the model's shards[]: self row uses local/in_vram/download
  // state directly; peer rows use holder_ids membership (disk if present, absent otherwise).
  function buildShardMatrix(m, shards, safeId, expanded) {
    if (!shards || shards.length === 0) return '';
    // Aggregate unique peers from holder_ids across all shards.
    var peerOrder = [];
    var peerIndex = {};
    shards.forEach(function(s) {
      (s.holder_ids || []).forEach(function(pid) {
        if (peerIndex[pid] === undefined) { peerIndex[pid] = peerOrder.length; peerOrder.push(pid); }
      });
    });
    // Count coverage per peer so we can sort by most complete first.
    var coverage = peerOrder.map(function(pid) {
      var c = 0;
      shards.forEach(function(s) { if ((s.holder_ids || []).indexOf(pid) !== -1) c++; });
      return { pid: pid, c: c };
    });
    coverage.sort(function(a, b) { return b.c - a.c; });
    var showAll = !!expanded;
    var capped = showAll ? coverage : coverage.slice(0, MATRIX_MAX_PEERS_DEFAULT);
    var overflow = coverage.length - capped.length;

    // Compute per-shard density tier so column headers can carry a mini
    // density bar above the index number — one unified view, no duplicate bar.
    var maxReplicas = 1;
    shards.forEach(function(s) {
      var total = (s.holders || 0);
      if (total > maxReplicas) maxReplicas = total;
    });
    var densityByIdx = {};
    shards.forEach(function(s) {
      var h = s.holders || 0;
      var pct = h === 0 ? 0 : Math.max(15, Math.round(Math.log(1 + h) / Math.log(1 + maxReplicas) * 100));
      var tier = h === 0 ? 'none' : h <= 2 ? 'low' : h <= 9 ? 'good' : 'high';
      densityByIdx[s.index] = { pct: pct, tier: tier, count: h };
    });

    // Column headers — mini density bar above the shard index number.
    // No leading label column; peer identity is carried by a colored left
    // border + hover tooltip on each row so shard columns align perfectly
    // with the coverage ribbon above.
    var headHtml = '<tr>';
    var colEvery = shards.length > 40 ? 5 : 1;
    var shardCountTotal = shards.length;
    shards.forEach(function(s, i) {
      var isMmproj = s.index === MMPROJ_SHARD_INDEX;
      var label = isMmproj ? '\u2605' : ((i % colEvery === 0) ? String(s.index + 1) : '');
      var d = densityByIdx[s.index] || { pct: 0, tier: 'none', count: 0 };
      var tip = I18n.t('shard.matrix.hist_col_tip', { n: s.index + 1, holders: d.count });
      // Mark endpoint columns so CSS + connector measurement can light them
      // up when the pipeline is E2E encrypted.
      var isFirst = shardCountTotal > 1 && s.index === 0;
      var isLast  = shardCountTotal > 1 && s.index === shardCountTotal - 1;
      // Mirrors list-view: pinned only when this node holds the endpoint
      // (the E2E guarantee needs local possession of first+last).
      var isPinned = (isFirst || isLast) && !!s.local && !!m.encrypted_pipeline;
      var thCls = [];
      if (isFirst) thCls.push('smh-endpoint-first');
      if (isLast)  thCls.push('smh-endpoint-last');
      if (isPinned) thCls.push('smh-col-pipeline-pinned');
      var thAttr = thCls.length ? ' class="' + thCls.join(' ') + '"' : '';
      headHtml += '<th' + thAttr + ' data-shard-col="' + s.index + '" title="' + U.escapeHtml(tip) + '">' +
        '<div class="smh-col" data-tier="' + d.tier + '">' +
          '<div class="smh-bar-wrap"><div class="smh-bar" style="height:' + d.pct + '%"></div></div>' +
          '<div class="smh-label">' + label + '</div>' +
        '</div>' +
        '</th>';
    });
    headHtml += '</tr>';

    // Self row — no leading <th>; left-border accent + title tooltip identifies.
    var selfRow = '<tr class="srm-row-self" title="' + U.escapeHtml(I18n.t('compare.filter_local')) + ' (' + U.escapeHtml(m.id) + ')">';
    shards.forEach(function(s) {
      var state = shardState(s);
      if (state === 'peer') state = 'absent';
      var glyph = state === 'vram' ? '\u25A0' : state === 'disk' ? '\u25A1' : '';
      var sIsFirst = shardCountTotal > 1 && s.index === 0;
      var sIsLast  = shardCountTotal > 1 && s.index === shardCountTotal - 1;
      var sPinned  = (sIsFirst || sIsLast) && !!s.local && !!m.encrypted_pipeline;
      var tdCls = sPinned ? ' class="smh-self-pipeline-pinned"' : '';
      selfRow += '<td' + tdCls + ' data-state="' + state + '" data-shard-col="' + s.index + '">' + glyph + '</td>';
    });
    selfRow += '</tr>';

    // Peer rows — colored left border (U.peerColor) + tooltip for identity.
    var peerRows = capped.map(function(entry) {
      var pid = entry.pid;
      var color = U.peerColor(pid);
      var row = '<tr class="srm-row-peer" data-peer-id="' + U.escapeHtml(pid) + '" style="--peer-color:' + color + '" title="' + U.escapeHtml(pid) + '">';
      shards.forEach(function(s) {
        var has = (s.holder_ids || []).indexOf(pid) !== -1;
        var state = has ? 'disk' : 'absent';
        var glyph = has ? '\u25A1' : '';
        row += '<td data-state="' + state + '" data-shard-col="' + s.index + '">' + glyph + '</td>';
      });
      row += '</tr>';
      return row;
    }).join('');

    var showAllBtn = overflow > 0
      ? '<button class="shard-matrix-showall" data-matrix-showall="' + safeId + '">' +
        U.escapeHtml(I18n.t('shard.matrix.show_all_peers', { n: coverage.length })) + '</button>'
      : '';

    var emptyHtml = (coverage.length === 0)
      ? '<div class="shard-matrix-empty">' + U.escapeHtml(I18n.t('shard.matrix.no_peers')) + '</div>'
      : '';

    return '<div class="shard-matrix" data-shard-matrix="' + safeId + '" data-shard-matrix-model="' + U.escapeHtml(m.id) + '"' + (showAll ? ' data-expanded="1"' : '') + '>' +
      '<div class="shard-matrix-wrap">' +
      '<table>' +
      '<thead>' + headHtml + '</thead>' +
      '<tbody>' + selfRow + peerRows + '</tbody>' +
      '</table>' +
      '<svg class="shard-matrix-path" data-matrix-path="' + safeId + '" aria-hidden="true"></svg>' +
      '</div>' +
      emptyHtml +
      showAllBtn +
      '</div>';
  }

  function buildShardDetailBody(m, shards, safeId) {
    var mode = S._shardView === 'matrix' ? 'matrix' : 'list';
    return mode === 'matrix'
      ? buildShardMatrix(m, shards, safeId, false)
      : buildShardList(m, shards, safeId);
  }

  /**
   * THE ROUTE STRIP — the model, drawn as the path a question takes through it.
   *
   * Left to right is not decoration: shard 0 turns your prompt into numbers and
   * the last shard writes the reply, so the strip reads in the order the work
   * actually happens. Colour says whose machine each piece is on, which is the
   * one fact a peer-to-peer dashboard has that an ordinary one does not — and
   * which the old 6px bar threw away by painting "yours" and "a stranger's" the
   * same shade of blue.
   *
   * It carried `ask` and `reply` end caps until 2026-09-13, removed on the
   * user's reading of them: they label the strip with a metaphor — a question
   * travelling through the model — that a reader scanning a model list is not
   * asking about, and the row now says in plain words what the strip shows
   * ("You host all 4 parts · 1.6 GB") with the colour key above it. The
   * left-to-right ORDER is still the order the work happens in; it simply no
   * longer announces itself twice per row.
   *
   * Segments stay equal-width rather than proportional to bytes: the
   * matrix columns below are equal-width, and a strip that disagreed with them
   * would be a worse lie than one that says nothing about size.
   *
   * Keeps the `availability-bar` class — `init.js` routes clicks on it to the
   * expand/collapse handler.
   */
  function buildCoverageRibbon(m, shards, safeId) {
    if (!shards || shards.length === 0) return '';
    var html = '<div class="availability-bar shard-coverage-ribbon route-strip" data-coverage-ribbon="' + safeId +
      '" title="' + U.escapeHtml(I18n.t('shard.view.coverage_tip') || '') + '">';
    shards.forEach(function(s) {
      var loc = shardLocality(s);
      var n = s.index === MMPROJ_SHARD_INDEX ? '★' : (s.index || 0) + 1;
      html += '<div class="avail-seg" data-loc="' + loc + '" title="' +
        U.escapeHtml(n + ' · ' + shardLocalityLabel(s, loc)) + '"></div>';
    });
    html += '</div>';
    return html;
  }

  /**
   * THE COLOUR KEY.
   *
   * Every colour in the route strip and the matrix meant something, and the
   * only way to find out what was to hover a segment. NN/g's icon guidance is
   * the same point the header pass acted on: don't rely on hover to carry a
   * label — it costs an interaction and does not exist on touch at all. A
   * peer-to-peer dashboard's one distinguishing fact is WHOSE machine each
   * piece is on, and it was the fact least available to a reader.
   *
   * It renders ONCE, above the model rows, and not inside each expanded card.
   * Placing it in the expanded view was details-on-demand applied to the wrong
   * thing: the strips a reader meets first are the COLLAPSED ones, every row
   * paints one, and those were exactly the strips the key could not reach —
   * while two open cards repeated the same six swatches at you. A key belongs
   * above the things it describes.
   *
   * The labels are the SAME `shard.loc.*` strings the hover titles use, so the
   * key cannot drift from what a segment says about itself. `swarm` is the one
   * exception: its own string is counted ("On {n} other computers"), which a
   * key cannot be.
   *
   * Every `data-loc` the strip can emit is listed, `moving` included. A colour
   * that appears in the strip and not in the key leaves the reader exactly
   * where the hover-only titles did, for that one state.
   */
  function buildShardLegend() {
    var items = [
      ['live', I18n.t('shard.loc.live')],
      ['disk', I18n.t('shard.loc.disk')],
      ['swarm', I18n.t('shard.legend.swarm')],
      ['thin', I18n.t('shard.loc.thin')],
      ['moving', I18n.t('shard.loc.moving')],
      ['absent', I18n.t('shard.loc.absent')],
    ];
    var html = '<div class="shard-legend"><span class="shard-legend-title">' +
      U.escapeHtml(I18n.t('shard.legend.title')) + '</span>';
    items.forEach(function(it) {
      html += '<span class="shard-legend-item">' +
        '<span class="avail-seg shard-legend-swatch" data-loc="' + it[0] + '"></span>' +
        U.escapeHtml(it[1]) + '</span>';
    });
    return html + '</div>';
  }

  App.dashboardShards = {
    MMPROJ_SHARD_INDEX: MMPROJ_SHARD_INDEX,
    buildProgressBar: buildProgressBar,
    shardLocality: shardLocality,
    shardLocalityLabel: shardLocalityLabel,
    shardState: shardState,
    shardWhereText: shardWhereText,
    buildPieceBar: buildPieceBar,
    buildRowActions: buildRowActions,
    buildShardRow: buildShardRow,
    buildShardList: buildShardList,
    buildShardViewToggle: buildShardViewToggle,
    buildShardLegend: buildShardLegend,
    buildShardMatrix: buildShardMatrix,
    buildShardDetailBody: buildShardDetailBody,
    buildCoverageRibbon: buildCoverageRibbon
  };
})();
