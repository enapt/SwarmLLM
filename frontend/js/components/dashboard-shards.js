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

  function shardGlyph(state) {
    // Filled square (▣), outlined square (▢), half-circle (◐ — peer fetching),
    // middle dot (·), heavy ballot (✕)
    return state === 'vram' ? '\u25A0'
         : state === 'disk' ? '\u25A1'
         : state === 'gossip' ? '\u25D0'
         : state === 'peer' ? '\u00B7'
         : '\u2715';
  }

  function shardStatusLabel(s, state) {
    if (state === 'vram') return I18n.t('shard.row.vram_label');
    if (state === 'disk') return I18n.t('dashboard.disk_label');
    if (state === 'gossip') {
      // Peer download in flight (gossip view). Show the leader's progress so
      // the user can see replication is moving.
      var lead = s.peer_downloads && s.peer_downloads[0]
        ? s.peer_downloads[0].progress_pct : 0;
      return (lead || 0) + '%\u2193';
    }
    if (state === 'peer') return I18n.t('shard.row.peer_label');
    return I18n.t('shard.row.missing_label');
  }

  // Compact replica indicator — single pill that scales to arbitrary N.
  // Tier: none=0, low=1-2, good=3-9, high=10+. Same layout irrespective of count.
  function shardReplicaPips(s) {
    // `holders` from the backend is the TOTAL count including self. The row
    // already visualizes the local-vs-remote dimension elsewhere, so this pip
    // surfaces just the remote replica count and says "Local only" when nobody
    // else has it.
    var holders = s.holders || 0;
    var isLocal = !!s.local;
    var others = Math.max(0, holders - (isLocal ? 1 : 0));
    var tier = others === 0 ? (isLocal ? 'local-only' : 'none')
             : others <= 2 ? 'low' : others <= 9 ? 'good' : 'high';
    var label, title;
    if (isLocal && others === 0) {
      label = '\u25C9'; // local-only glyph (filled circle)
      title = I18n.t('shard.row.replicas_local_only');
    } else if (others === 0) {
      label = '\u2014';
      title = I18n.t('shard.row.replicas_none');
    } else {
      label = (isLocal ? '+' : '') + String(others);
      title = I18n.t(
        isLocal ? 'shard.row.replicas_local_plus' : (others === 1 ? 'shard.row.replicas_count_one' : 'shard.row.replicas_count_other'),
        { n: others }
      );
    }
    return '<span class="shard-row-replicas" data-tier="' + tier + '"' + (isLocal ? ' data-local="1"' : '') + ' title="' + U.escapeHtml(title) + '">' +
      '<span class="shard-row-replica-dot"></span>' +
      '<span class="shard-row-replica-count">' + label + '</span>' +
      '</span>';
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

  function buildShardRow(s, m, safeId) {
    var state = shardState(s);
    var isMmproj = s.index === MMPROJ_SHARD_INDEX;
    var idxLabel = isMmproj ? '\u2605' : String((s.index || 0) + 1);
    var shardCount = m.shard_count || (m.shards || []).length || 0;
    var isFirst = shardCount > 1 && s.index === 0;
    var isLast  = shardCount > 1 && s.index === shardCount - 1;
    var isEndpoint = isFirst || isLast;
    var isPipelinePinned = isEndpoint && s.local && m.encrypted_pipeline;
    var endpointBadge = '';
    if (isFirst) {
      endpointBadge = '<span class="shard-row-endpoint" data-kind="first" title="' + U.escapeHtml(I18n.t('shard.endpoint_first_tip')) + '">' + U.escapeHtml(I18n.t('shard.endpoint_first')) + '</span>';
    } else if (isLast) {
      endpointBadge = '<span class="shard-row-endpoint" data-kind="last" title="' + U.escapeHtml(I18n.t('shard.endpoint_last_tip')) + '">' + U.escapeHtml(I18n.t('shard.endpoint_last')) + '</span>';
    }
    // The user deleted this piece from this device and has not asked for it
    // since: auto-manage will not bring it back on its own (external report,
    // 2026-08-21 — a deliberate two-machine split was silently undone).
    var removedBadge = (s.removed_by_user && !s.local)
      ? '<span class="shard-row-endpoint" data-kind="removed" title="' + U.escapeHtml(I18n.t('shard.removed_by_user_tip')) + '">' + U.escapeHtml(I18n.t('shard.removed_by_user')) + '</span>'
      : '';
    var layerRange = '';
    var statusLabel = shardStatusLabel(s, state);
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
      '<span class="shard-row-state-glyph">' + shardGlyph(state) + '</span>' +
      '<span class="shard-row-index">' + idxLabel + endpointBadge + removedBadge + '</span>' +
      '<span class="shard-row-layers">' + layerRange + '</span>' +
      '<span class="shard-row-status">' + U.escapeHtml(statusLabel) + '</span>' +
      shardReplicaPips(s) +
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
   * The `ask`/`reply` end caps appear in compact mode only. Expanded mode
   * labels its own endpoints on the shard rows (`shard-row-endpoint`), and
   * keeping the caps out of it leaves the strip's geometry alone next to the
   * matrix. Segments stay equal-width rather than proportional to bytes: the
   * matrix columns below are equal-width, and a strip that disagreed with them
   * would be a worse lie than one that says nothing about size.
   *
   * Keeps the `availability-bar` class — `init.js` routes clicks on it to the
   * expand/collapse handler.
   */
  function buildCoverageRibbon(m, shards, safeId) {
    if (!shards || shards.length === 0) return '';
    var multi = shards.length > 1;
    var html = '<div class="availability-bar shard-coverage-ribbon route-strip" data-coverage-ribbon="' + safeId +
      '" title="' + U.escapeHtml(I18n.t('shard.view.coverage_tip') || '') + '">';
    if (multi) {
      html += '<span class="route-cap route-cap-in" aria-hidden="true">' +
        U.escapeHtml(I18n.t('shard.route.ask')) + '</span>';
    }
    shards.forEach(function(s) {
      var loc = shardLocality(s);
      var n = s.index === MMPROJ_SHARD_INDEX ? '★' : (s.index || 0) + 1;
      html += '<div class="avail-seg" data-loc="' + loc + '" title="' +
        U.escapeHtml(n + ' · ' + shardLocalityLabel(s, loc)) + '"></div>';
    });
    if (multi) {
      html += '<span class="route-cap route-cap-out" aria-hidden="true">' +
        U.escapeHtml(I18n.t('shard.route.reply')) + '</span>';
    }
    html += '</div>';
    return html;
  }

  App.dashboardShards = {
    MMPROJ_SHARD_INDEX: MMPROJ_SHARD_INDEX,
    buildProgressBar: buildProgressBar,
    shardLocality: shardLocality,
    shardLocalityLabel: shardLocalityLabel,
    shardState: shardState,
    shardGlyph: shardGlyph,
    shardStatusLabel: shardStatusLabel,
    shardReplicaPips: shardReplicaPips,
    buildPieceBar: buildPieceBar,
    buildRowActions: buildRowActions,
    buildShardRow: buildShardRow,
    buildShardList: buildShardList,
    buildShardViewToggle: buildShardViewToggle,
    buildShardMatrix: buildShardMatrix,
    buildShardDetailBody: buildShardDetailBody,
    buildCoverageRibbon: buildCoverageRibbon
  };
})();
