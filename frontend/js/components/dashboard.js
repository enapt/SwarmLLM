'use strict';

// ============================================================================
// SwarmLLM — Dashboard Component
// Stats, model cards, peer list, shard grid, acquisition progress
// ============================================================================

(function() {
  var S = App.state;
  var U = App.utils;

  // Per-model event logs — populated from backend activity_history replay on WS connect
  var _modelEvents = {};
  var _modelNetEvents = {};

  // Kinds that go to the network ticker on model cards
  var MODEL_NET_KINDS = { 'shard_announced': 1, 'peer_connected': 1, 'peer_disconnected': 1, 'rebalance_peer_left': 1 };

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
  function _buildProgressBar(opts) {
    var bar = opts.barContent || '<div class="dl-fill" style="width:' + opts.pct + '%"></div>';
    var right = opts.cancelBtn
      ? '<span style="display:flex;align-items:center;gap:8px"><span class="mono dl-progress-text">' + opts.rightText + '</span>' + opts.cancelBtn + '</span>'
      : '<span class="mono dl-progress-text">' + opts.rightText + '</span>';
    return '<div class="dl-progress" data-model-progress="' + opts.safeId + '" data-last-pct="' + opts.pct + '">' +
      '<div class="flex-between field-hint mb-0">' +
      '<span class="text-muted">' + opts.label + '</span>' +
      right +
      '</div>' +
      '<div class="dl-bar">' + bar + '</div>' +
      '</div>';
  }

  // ==========================================================================
  // Shard row list — new dense row-per-shard rendering (replaces grid)
  // ==========================================================================
  var MMPROJ_SHARD_INDEX = 0xFFFFFFFF;

  function _shardState(s) {
    if (s.download && s.download.state === 'Downloading') return 'downloading';
    if (s.download && s.download.state === 'Verifying') return 'downloading';
    if (s.download && (s.download.state === 'Queued' || s.download.state === 'pending')) return 'downloading';
    if (s.peer_downloads && s.peer_downloads.length > 0 && !s.local) return 'downloading';
    if (s.local && s.in_vram) return 'vram';
    if (s.local) return 'disk';
    if ((s.holders || 0) > 0) return 'peer';
    return 'missing';
  }

  function _shardGlyph(state) {
    // Filled square (▣), outlined square (▢), half-circle (◐), middle dot (·), heavy ballot (✕)
    return state === 'vram' ? '\u25A0'
         : state === 'disk' ? '\u25A1'
         : state === 'downloading' ? '\u25D0'
         : state === 'peer' ? '\u00B7'
         : '\u2715';
  }

  function _shardStatusLabel(s, state) {
    if (state === 'vram') return I18n.t('shard.row.vram_label');
    if (state === 'disk') return I18n.t('shard.row.disk_label');
    if (state === 'downloading') {
      var pct = (s.download && typeof s.download.progress_pct === 'number')
        ? s.download.progress_pct
        : (s.peer_downloads && s.peer_downloads[0] ? s.peer_downloads[0].progress_pct : 0);
      return (pct || 0) + '%\u2193';
    }
    if (state === 'peer') return I18n.t('shard.row.peer_label');
    return I18n.t('shard.row.missing_label');
  }

  // Compact replica indicator — single pill that scales to arbitrary N.
  // Tier: none=0, low=1-2, good=3-9, high=10+. Same layout irrespective of count.
  function _shardReplicaPips(s) {
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
  function _buildPieceBar(peerDownloads, totalPct) {
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

  function _buildRowActions(state, isLocal, isInVram) {
    var parts = [];
    if (state === 'disk') {
      parts.push('<button class="shard-row-act" data-shard-act="load" title="' + U.escapeHtml(I18n.t('shard.row.action_load')) + '">\u25B2</button>');
    } else if (state === 'vram') {
      parts.push('<button class="shard-row-act" data-shard-act="unload" title="' + U.escapeHtml(I18n.t('shard.row.action_unload')) + '">\u25BC</button>');
    }
    if (state === 'downloading') {
      parts.push('<button class="shard-row-act danger" data-shard-act="cancel" title="' + U.escapeHtml(I18n.t('shard.row.action_cancel')) + '">\u2715</button>');
    } else if (state === 'peer' || state === 'missing') {
      parts.push('<button class="shard-row-act" data-shard-act="download" title="' + U.escapeHtml(I18n.t('shard.row.action_download')) + '">\u21E9</button>');
    }
    if (isLocal && state !== 'downloading') {
      parts.push('<button class="shard-row-act danger" data-shard-act="delete" title="' + U.escapeHtml(I18n.t('shard.row.action_delete')) + '">\u2302</button>');
    }
    return parts.length ? '<span class="shard-row-actions">' + parts.join('') + '</span>' : '';
  }

  function _buildShardRow(s, m, safeId) {
    var state = _shardState(s);
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
    var layerRange = '';
    var statusLabel = _shardStatusLabel(s, state);
    var sizeText = s.size_bytes ? U.formatBytes(s.size_bytes) : '\u2014';
    var lockCls = s.locked ? ' locked' : '';
    // Pushpin icon = "pin to device" (auto-manage). Reserved 🔒/🔓 for pipeline encryption.
    var lockGlyph = '\uD83D\uDCCC';
    var lockTitle = s.locked ? I18n.t('shard.unlock') : I18n.t('shard.lock');
    var pieceBar = (state === 'downloading' && s.peer_downloads && s.peer_downloads.length > 0)
      ? _buildPieceBar(s.peer_downloads, (s.download && s.download.progress_pct) || 0)
      : '';
    var actions = _buildRowActions(state, !!s.local, !!s.in_vram);
    var rowClass = 'shard-row';
    if (isEndpoint) rowClass += ' shard-row-endpoint-row';
    if (isPipelinePinned) rowClass += ' shard-row-pipeline-pinned';
    return '<div class="' + rowClass + '" data-state="' + state + '"' +
      ' data-shard-row="' + safeId + '-' + s.index + '"' +
      ' data-shard-model="' + U.escapeHtml(m.id) + '"' +
      ' data-shard-index="' + s.index + '"' +
      ' data-shard-locked="' + (s.locked ? '1' : '0') + '">' +
      '<span class="shard-row-state-glyph">' + _shardGlyph(state) + '</span>' +
      '<span class="shard-row-index">' + idxLabel + endpointBadge + '</span>' +
      '<span class="shard-row-layers">' + layerRange + '</span>' +
      '<span class="shard-row-status">' + U.escapeHtml(statusLabel) + '</span>' +
      _shardReplicaPips(s) +
      '<span class="shard-row-size">' + sizeText + '</span>' +
      '<button class="shard-row-lock' + lockCls + '" data-shard-act="toggle-lock" title="' + U.escapeHtml(lockTitle) + '">' + lockGlyph + '</button>' +
      '<button class="shard-row-more" data-shard-act="expand" title="' + U.escapeHtml(I18n.t('shard.row.expand_tip')) + '">\u203A</button>' +
      actions +
      pieceBar +
      '</div>';
  }

  function _buildShardList(m, shards, safeId) {
    if (!shards || shards.length === 0) return '';
    var rows = shards.map(function(s) { return _buildShardRow(s, m, safeId); }).join('');
    return '<div class="shard-list" data-shard-list="' + safeId + '">' + rows + '</div>';
  }

  function _buildShardViewToggle() {
    var mode = S._shardView === 'matrix' ? 'matrix' : 'list';
    return '<div class="shard-view-toggle" role="tablist">' +
      '<button type="button" data-shard-view="list" class="' + (mode === 'list' ? 'active' : '') + '" title="' + U.escapeHtml(I18n.t('shard.view.toggle_tip') || '') + '">' + U.escapeHtml(I18n.t('shard.view.list')) + '</button>' +
      '<button type="button" data-shard-view="matrix" class="' + (mode === 'matrix' ? 'active' : '') + '" title="' + U.escapeHtml(I18n.t('shard.view.toggle_tip') || '') + '">' + U.escapeHtml(I18n.t('shard.view.matrix')) + '</button>' +
      '</div>';
  }

  // Matrix view — rows = peers (self pinned top), cols = shards.
  // Cell state derived from the model's shards[]: self row uses local/in_vram/download
  // state directly; peer rows use holder_ids membership (disk if present, absent otherwise).
  var MATRIX_MAX_PEERS_DEFAULT = 12;

  function _buildShardMatrix(m, shards, safeId, expanded) {
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
    var selfRow = '<tr class="srm-row-self" title="' + U.escapeHtml(I18n.t('shard.matrix.peer_you')) + ' (' + U.escapeHtml(m.id) + ')">';
    shards.forEach(function(s, i) {
      var state = _shardState(s);
      if (state === 'peer') state = 'absent';
      var glyph = state === 'vram' ? '\u25A0' : state === 'disk' ? '\u25A1'
                : state === 'downloading' ? '\u25D0' : '';
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

  function _buildShardDetailBody(m, shards, safeId) {
    var mode = S._shardView === 'matrix' ? 'matrix' : 'list';
    return mode === 'matrix'
      ? _buildShardMatrix(m, shards, safeId, false)
      : _buildShardList(m, shards, safeId);
  }

  // Coverage ribbon for the expanded panel's right column — a compact strip
  // colored by network replica count per shard. Reuses availability-bar semantics.
  function _buildCoverageRibbon(m, shards, safeId) {
    if (!shards || shards.length === 0) return '';
    var html = '<div class="availability-bar shard-coverage-ribbon" data-coverage-ribbon="' + safeId +
      '" title="' + U.escapeHtml(I18n.t('shard.view.coverage_tip') || '') + '">';
    shards.forEach(function(s) {
      var segClass = 'seg-missing';
      if (s.local && s.in_vram) segClass = 'seg-active';
      else if (s.local) segClass = 'seg-nominal';
      else if (s.download && (s.download.state === 'Downloading' || s.download.state === 'Verifying')) segClass = 'seg-downloading';
      else if (s.peer_downloads && s.peer_downloads.length > 0) segClass = 'seg-downloading';
      else if ((s.holders || 0) >= 2) segClass = 'seg-peer';
      else if ((s.holders || 0) === 1) segClass = 'seg-warning';
      else segClass = 'seg-problem';
      html += '<div class="avail-seg ' + segClass + '"></div>';
    });
    html += '</div>';
    return html;
  }


  App.dashboard = {
    _peersExpanded: false,
    _lastPeers: [],

    // Swap all expanded model cards' right-column bodies between list and matrix.
    // Called by the delegated click handler on .shard-view-toggle buttons.
    setShardView: function(mode) {
      if (mode !== 'list' && mode !== 'matrix') return;
      S._shardView = mode;
      try { localStorage.setItem(App.SHARD_VIEW_KEY, mode); } catch (e) {}
      var cached = (App.data && App.data.cache && App.data.cache.models) || [];
      var byId = {};
      cached.forEach(function(m) { byId[m.id] = m; });
      document.querySelectorAll('[data-shard-detail]').forEach(function(rightEl) {
        var safeId = rightEl.getAttribute('data-shard-detail');
        var card = rightEl.closest('.model-card');
        var modelId = card ? card.getAttribute('data-model-id') : null;
        var model = modelId ? byId[modelId] : null;
        if (!model) return;
        var body = rightEl.querySelector('.mce-right-body');
        if (body) body.innerHTML = _buildShardDetailBody(model, model.shards || [], safeId);
        // Update toggle active states
        rightEl.querySelectorAll('.shard-view-toggle button').forEach(function(btn) {
          var v = btn.getAttribute('data-shard-view');
          if (v === mode) btn.classList.add('active'); else btn.classList.remove('active');
        });
        // Re-measure pipeline connector since the anchors changed
        // (rows ↔ columns) with the view switch.
        if (card) {
          requestAnimationFrame(function() {
            App.dashboard._measurePipelineConnector(card);
            App.dashboard._applyPipelinePlan(card);
          });
        }
      });
    },

    // Inline shard-row action dispatcher. Maps each data-shard-act value to its
    // existing per-shard API endpoint. Replaces the old shard-menu.js popup flow.
    shardRowAction: function(action, modelId, shardIndex, rowEl) {
      var url = U.modelApiUrl(modelId, 'shards', shardIndex);
      var rowReload = function() { App.models.load(); };

      if (action === 'toggle-lock') {
        var wasLocked = rowEl && rowEl.getAttribute('data-shard-locked') === '1';
        var newLocked = !wasLocked;
        App.authFetch(url + '/lock', {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ locked: newLocked }),
        }).then(function(resp) {
          if (resp.ok) {
            App.notifications.showToast(I18n.t(newLocked ? 'shard.locked' : 'shard.unlocked', { idx: shardIndex + 1 }), 'success');
            rowReload();
          } else {
            App.notifications.showToast(I18n.t('shard.lock_failed'), 'error');
          }
        }).catch(function(err) {
          App.notifications.showToast(I18n.t('shard.lock_error', { error: err.message }), 'error');
        });
        return;
      }

      if (action === 'expand') {
        App.dashboard.toggleShardRowExpand(rowEl);
        return;
      }

      if (action === 'load') {
        App.authFetch(url + '/load', { method: 'POST' }).then(function(resp) {
          return resp.ok
            ? App.notifications.showToast(I18n.t('shard.loading', { idx: shardIndex + 1 }), 'success')
            : U.getApiErrorMessage(resp, I18n.t('shard.load_failed')).then(function(msg) { App.notifications.showToast(msg, 'error'); });
        }).then(rowReload).catch(function(err) {
          App.notifications.showToast(I18n.t('shard.load_error', { error: err.message }), 'error');
        });
        return;
      }

      if (action === 'unload') {
        if (!confirm(I18n.t('shard.confirm_unload', { idx: shardIndex + 1 }))) return;
        App.authFetch(url + '/unload', { method: 'POST' }).then(function(resp) {
          if (resp.ok) {
            var name = U.formatModelDisplayName(modelId);
            App.notifications.showToast(I18n.t('shard.unloaded', { idx: shardIndex + 1, model: name }), 'success');
            rowReload();
          } else {
            U.getApiErrorMessage(resp, I18n.t('shard.unload_failed')).then(function(msg) { App.notifications.showToast(msg, 'error'); });
          }
        }).catch(function(err) {
          App.notifications.showToast(I18n.t('shard.unload_error', { error: err.message }), 'error');
        });
        return;
      }

      if (action === 'delete') {
        if (!confirm(I18n.t('actions.confirm_remove_shard', { index: shardIndex + 1, model: modelId }))) return;
        App.authFetch(url, { method: 'DELETE' }).then(function(resp) {
          if (resp.ok) {
            App.ui.showBanner('success', I18n.t('shard.removed', { idx: shardIndex + 1 }));
            rowReload();
          } else {
            U.getApiErrorMessage(resp, I18n.t('shard.remove_failed')).then(function(msg) { App.ui.showBanner('error', msg); });
          }
        }).catch(function(err) {
          App.ui.showBanner('error', I18n.t('shard.remove_error', { error: err.message }));
        });
        return;
      }

      if (action === 'cancel') {
        App.models.cancelDownload(modelId);
        return;
      }

      if (action === 'download') {
        App.authFetch(url + '/download', { method: 'POST' }).then(function(resp) {
          return resp.json();
        }).then(function(data) {
          if (data.status === 'downloading') {
            App.ui.showBanner('success', I18n.t('shard.downloading_from', {
              idx: shardIndex + 1,
              source: data.source === 'p2p' ? I18n.t('shard.source_peer', { id: data.peer || '' }) : I18n.t('shard.source_peers'),
            }));
            rowReload();
          } else if (data.status === 'use_hf') {
            App.hf.downloadShards({ repo_id: data.repo_id, filename: data.filename, shards: [shardIndex], model_id: modelId }).then(function(r) {
              if (r.ok) {
                App.ui.showBanner('success', I18n.t('shard.downloading_hf', { idx: shardIndex + 1 }));
                rowReload();
              } else {
                App.ui.showBanner('error', r.errorMsg || I18n.t('shard.hf_download_failed'));
              }
            });
          } else if (data.status === 'already_local') {
            App.ui.showBanner('info', I18n.t('shard.already_local', { idx: shardIndex + 1 }));
          } else {
            App.ui.showBanner('error', U.extractErrorMessage(data, I18n.t('shard.download_unavailable')));
          }
        }).catch(function(err) {
          App.ui.showBanner('error', I18n.t('shard.download_failed', { error: err.message }));
        });
        return;
      }
    },

    // Accordion expand/collapse. Only one row per model expanded at a time.
    toggleShardRowExpand: function(rowEl) {
      if (!rowEl) return;
      var isExpanded = rowEl.classList.contains('expanded');
      var list = rowEl.parentElement;
      if (list) {
        list.querySelectorAll('.shard-row.expanded').forEach(function(other) {
          if (other !== rowEl) {
            other.classList.remove('expanded');
            var panel = other.querySelector('.shard-row-expanded-panel');
            if (panel) panel.remove();
          }
        });
      }
      if (isExpanded) {
        rowEl.classList.remove('expanded');
        var ep = rowEl.querySelector('.shard-row-expanded-panel');
        if (ep) ep.remove();
        return;
      }
      // Build detail panel using cached model data
      var modelId = rowEl.getAttribute('data-shard-model');
      var idx = parseInt(rowEl.getAttribute('data-shard-index'), 10);
      var cached = (App.data && App.data.cache && App.data.cache.models) || [];
      var model = null;
      for (var i = 0; i < cached.length; i++) { if (cached[i].id === modelId) { model = cached[i]; break; } }
      if (!model) return;
      var shard = (model.shards || []).find(function(s) { return s.index === idx; });
      if (!shard) return;

      var state = _shardState(shard);
      var holders = shard.holder_ids || [];
      var holdersHtml = holders.length === 0
        ? '<span class="text-muted">' + U.escapeHtml(I18n.t('shard.row.no_other_holders')) + '</span>'
        : holders.slice(0, 16).map(function(pid) {
            var short = pid.length > 12 ? pid.substring(0, 12) : pid;
            return '<span class="srep-holder-chip"><span class="srep-holder-swatch" style="background:' + U.peerColor(pid) + '"></span>' + U.escapeHtml(short) + '</span>';
          }).join('');

      var destructive = [];
      if (state === 'disk') destructive.push('<button data-shard-act="load">' + U.escapeHtml(I18n.t('shard.row.action_load')) + '</button>');
      if (state === 'vram') destructive.push('<button data-shard-act="unload">' + U.escapeHtml(I18n.t('shard.row.action_unload')) + '</button>');
      if (state === 'downloading') destructive.push('<button class="danger" data-shard-act="cancel">' + U.escapeHtml(I18n.t('shard.row.action_cancel')) + '</button>');
      if (state === 'peer' || state === 'missing') destructive.push('<button data-shard-act="download">' + U.escapeHtml(I18n.t('shard.row.action_download')) + '</button>');
      if (shard.local && state !== 'downloading') destructive.push('<button class="danger" data-shard-act="delete">' + U.escapeHtml(I18n.t('shard.row.action_delete')) + '</button>');

      var panelHtml = '<div class="shard-row-expanded-panel">' +
        '<div class="srep-section">' +
          '<div class="srep-section-label">' + U.escapeHtml(I18n.t('shard.row.holders_title')) + '</div>' +
          '<div class="srep-holders">' + holdersHtml + '</div>' +
        '</div>' +
        (shard.size_bytes ? '<div class="srep-section"><span class="srep-section-label">Size</span> ' + U.formatBytes(shard.size_bytes) + '</div>' : '') +
        '<div class="srep-destructive">' + destructive.join('') + '</div>' +
        '</div>';
      rowEl.insertAdjacentHTML('beforeend', panelHtml);
      rowEl.classList.add('expanded');
    },

    expandMatrixAllPeers: function(safeId) {
      var mx = document.querySelector('[data-shard-matrix="' + U.cssSafeAttr(safeId) + '"]');
      if (!mx) return;
      var card = mx.closest('.model-card');
      var modelId = card ? card.getAttribute('data-model-id') : null;
      var cached = (App.data && App.data.cache && App.data.cache.models) || [];
      var model = null;
      for (var i = 0; i < cached.length; i++) { if (cached[i].id === modelId) { model = cached[i]; break; } }
      if (!model) return;
      var body = mx.parentElement;
      if (body) body.innerHTML = _buildShardMatrix(model, model.shards || [], safeId, true);
    },

    _logModelEvent: function(modelId, icon, text, skipGlobal, kind) {
      var isNet = kind && MODEL_NET_KINDS[kind];
      var store = isNet ? _modelNetEvents : _modelEvents;
      if (!store[modelId]) store[modelId] = [];
      var events = store[modelId];
      var ts = Date.now();
      events.unshift({ icon: icon, text: text, ts: ts });
      if (events.length > 15) events.pop();

      App.dashboard._renderModelTicker(modelId);

      // Also log to global panel (unless the caller already did via activity_event)
      if (!skipGlobal) {
        App.notifications.logActivity(icon, U.formatModelDisplayName(modelId) + ': ' + text, isNet ? 'network' : 'model', modelId);
      }
    },

    // Render the per-model ticker DOM — split into activity + network columns
    _updateContribution: function(pct, memKind) {
      var el = document.getElementById('contribution-pct');
      if (!el) return;
      var tier = pct < 5 ? 'idle' : pct < 25 ? 'minimal' : pct < 60 ? 'moderate' : 'maximum';
      var tierLabel = I18n.t('dashboard.contribution_tier_' + tier) || tier;
      var memLabel = memKind === 'vram' ? I18n.t('hw.vram') : I18n.t('hw.ram');
      el.textContent = pct.toFixed(0) + '% ' + memLabel + ' · ' + tierLabel;
    },

    // Measure pinned endpoint shard rows and set CSS vars on .mce-right so
    // the pipeline-encrypted connector line spans exactly from the first to
    // the last endpoint tick. Safe to call repeatedly.
    _measurePipelineConnector: function(card) {
      if (!card) return;
      var exp = card.querySelector('.model-card-expanded.pipeline-encrypted');
      var right = exp && exp.querySelector('.mce-right');
      if (!right) return;
      // List view uses shard rows; matrix view anchors to the self row's
      // first+last cells (where this node holds the pipeline endpoints),
      // not the column headers (which show density across all peers).
      var pinned = right.querySelectorAll('.shard-row-pipeline-pinned');
      var isMatrix = false;
      if (pinned.length < 1) {
        pinned = right.querySelectorAll('.smh-self-pipeline-pinned');
        isMatrix = pinned.length > 0;
      }
      right.classList.toggle('pipe-matrix', isMatrix);
      if (pinned.length < 1) {
        right.style.removeProperty('--pipe-line-top');
        right.style.removeProperty('--pipe-line-bottom');
        right.style.removeProperty('--pipe-tail-x');
        return;
      }
      // Line must span all three connection points: the privacy panel's
      // stub (at its vertical center) + the first and last pinned rows.
      // With 2 shards, first == last, so without the stub anchor the line
      // would collapse to a single row and not reach the privacy panel.
      var rightRect = right.getBoundingClientRect();
      var firstRect = pinned[0].getBoundingClientRect();
      var lastRect  = pinned[pinned.length - 1].getBoundingClientRect();
      var anchors = [
        (firstRect.top + firstRect.height / 2) - rightRect.top,
        (lastRect.top  + lastRect.height  / 2) - rightRect.top,
      ];
      var privacy = exp.querySelector('.mce-section-privacy');
      if (privacy) {
        // Stub sits at the privacy panel's vertical center (matches CSS).
        var pRect = privacy.getBoundingClientRect();
        anchors.push((pRect.top + pRect.height / 2) - rightRect.top);
      }
      var topOffset    = Math.min.apply(null, anchors);
      var bottomOffset = rightRect.height - Math.max.apply(null, anchors);
      right.style.setProperty('--pipe-line-top', topOffset + 'px');
      right.style.setProperty('--pipe-line-bottom', bottomOffset + 'px');
      // Matrix view: add a horizontal tail from the line across to the
      // first pinned column so the visual connection is obvious.
      if (isMatrix) {
        var firstPinnedRect = pinned[0].getBoundingClientRect();
        var tailTopY = (firstPinnedRect.top + firstPinnedRect.height / 2) - rightRect.top;
        var tailRightX = rightRect.right - (firstPinnedRect.left + firstPinnedRect.width / 2);
        right.style.setProperty('--pipe-tail-y', tailTopY + 'px');
        right.style.setProperty('--pipe-tail-x', tailRightX + 'px');
      } else {
        right.style.removeProperty('--pipe-tail-y');
        right.style.removeProperty('--pipe-tail-x');
      }
    },
    // Fetch the scheduler's pipeline plan for this model and render the
    // inference path on top of the shard matrix: mark chosen peer+shard
    // cells and draw an SVG polyline connecting them in segment order.
    // Unchosen holders are dimmed so the path stands out.
    _applyPipelinePlan: function(card) {
      if (!card) return;
      var modelId = card.getAttribute('data-model-id');
      if (!modelId) return;
      var matrix = card.querySelector('[data-shard-matrix]');
      if (!matrix) return;
      var table = matrix.querySelector('table');
      var svg = matrix.querySelector('.shard-matrix-path');
      if (!table || !svg) return;
      // Clear previous plan state
      matrix.removeAttribute('data-has-plan');
      matrix.querySelectorAll('.planned-cell').forEach(function(el) { el.classList.remove('planned-cell'); });
      matrix.querySelectorAll('.planned-row').forEach(function(el) { el.classList.remove('planned-row'); });
      svg.innerHTML = '';

      App.authFetch('/api/admin/models/' + encodeURIComponent(modelId) + '/pipeline-plan')
        .then(function(res) { return res.ok ? res.json() : null; })
        .then(function(plan) {
          if (!plan || !plan.segments || plan.segments.length === 0) return;
          var localId = plan.local_node_id;
          matrix.setAttribute('data-has-plan', '1');
          matrix.querySelectorAll('tbody tr').forEach(function(tr) { tr.classList.add('unplanned-row'); });

          var points = [];
          plan.segments.forEach(function(seg, i) {
            var peerId = seg.node_id;
            var row = peerId === localId
              ? matrix.querySelector('tr.srm-row-self')
              : matrix.querySelector('tr.srm-row-peer[data-peer-id="' + U.cssSafeAttr(peerId) + '"]');
            if (!row) return;
            row.classList.remove('unplanned-row');
            row.classList.add('planned-row');
            var td = row.querySelector('td[data-shard-col="' + seg.shard_index + '"]');
            if (!td) return;
            td.classList.add('planned-cell');
            td.setAttribute('data-plan-order', String(i + 1));
            var tblRect = table.getBoundingClientRect();
            var r = td.getBoundingClientRect();
            points.push({
              x: (r.left + r.width / 2) - tblRect.left,
              y: (r.top + r.height / 2) - tblRect.top,
              local: seg.is_local,
            });
          });
          if (points.length < 1) return;

          var w = table.clientWidth;
          var h = table.clientHeight;
          svg.setAttribute('viewBox', '0 0 ' + w + ' ' + h);
          svg.setAttribute('width', w);
          svg.setAttribute('height', h);
          var ns = 'http://www.w3.org/2000/svg';
          var d = points.map(function(p, i) { return (i === 0 ? 'M' : 'L') + p.x + ' ' + p.y; }).join(' ');
          var path = document.createElementNS(ns, 'path');
          path.setAttribute('d', d);
          path.setAttribute('class', 'shard-matrix-path-line');
          svg.appendChild(path);
          points.forEach(function(p, i) {
            var c = document.createElementNS(ns, 'circle');
            c.setAttribute('cx', p.x);
            c.setAttribute('cy', p.y);
            c.setAttribute('r', '5');
            c.setAttribute('class', 'shard-matrix-path-dot' + (p.local ? ' local' : ''));
            svg.appendChild(c);
            var t = document.createElementNS(ns, 'text');
            t.setAttribute('x', p.x);
            t.setAttribute('y', p.y + 3);
            t.setAttribute('class', 'shard-matrix-path-label');
            t.textContent = String(i + 1);
            svg.appendChild(t);
          });
        })
        .catch(function() { /* quiet: plan unavailable (no peers etc.) */ });
    },
    _renderModelTicker: function(modelId) {
      var actEvents = _modelEvents[modelId] || [];
      var netEvents = _modelNetEvents[modelId] || [];
      if (actEvents.length === 0 && netEvents.length === 0) return;

      var safeId = U.safeId(modelId);
      var ticker = document.querySelector('[data-model-ticker="' + safeId + '"]');
      if (!ticker) return;

      function _tickerTime(ts) {
        var d = new Date(ts);
        return ('0' + d.getHours()).slice(-2) + ':' + ('0' + d.getMinutes()).slice(-2) + ':' + ('0' + d.getSeconds()).slice(-2);
      }
      function _renderColumn(events, emptyText) {
        if (events.length === 0) return '<div class="text-muted text-2xs py-1">' + emptyText + '</div>';
        var latest = events[0];
        var html = '<div class="model-ticker-latest"><span class="model-ticker-icon">' + latest.icon + '</span>' +
          '<span class="model-ticker-text">' + U.escapeHtml(latest.text) + '</span>' +
          '<span class="model-ticker-time" data-ts="' + latest.ts + '">' + U.timeAgo(latest.ts) + '</span></div>';
        if (events.length > 1) {
          html += '<div class="model-ticker-history">';
          events.slice(1, 6).forEach(function(e) {
            html += '<div class="model-ticker-row"><span>' + e.icon + ' ' + U.escapeHtml(e.text) + '</span><span class="model-ticker-time" data-ts="' + e.ts + '">' + _tickerTime(e.ts) + ' ' + U.timeAgo(e.ts) + '</span></div>';
          });
          html += '</div>';
        }
        return html;
      }

      ticker.innerHTML =
        '<div class="model-ticker-split">' +
          '<div class="model-ticker-col"><div class="model-ticker-col-label">' + I18n.t('activity.label_activity') + '</div>' + _renderColumn(actEvents, I18n.t('activity.none')) + '</div>' +
          '<div class="model-ticker-col"><div class="model-ticker-col-label">' + I18n.t('activity.label_network') + '</div>' + _renderColumn(netEvents, I18n.t('activity.none_network')) + '</div>' +
        '</div>';
      ticker.style.display = '';
    },

    loadInitial: async function() {
      // Debounce: skip if already loading or loaded within 5s
      if (App.dashboard._loading) return;
      var now = Date.now();
      if (now - (App.dashboard._lastLoadTime || 0) < 5000) return;
      App.dashboard._loading = true;
      App.dashboard._lastLoadTime = now;
      var statsResult;
      try {
        var results = await Promise.all([App.data.loadStats(), App.models.load()]);
        statsResult = results[0];
      } catch (e) {
        App.dashboard._loading = false;
        App.ui.showBanner('error', I18n.t('errors.server_unreachable'));
        return;
      }

      if (statsResult.stats) {
        App.dashboard.updateFull(statsResult.stats);
      } else {
        App.ui.showBanner('error', I18n.t('errors.server_unreachable'));
      }

      if (statsResult.config) {
        var cfg = statsResult.config;
        if (cfg.contribution) document.getElementById('settings-contribution').value = cfg.contribution;
        if (cfg.max_concurrent_requests) document.getElementById('settings-max-requests').value = cfg.max_concurrent_requests;
        if (cfg.max_bandwidth_mbps !== undefined) document.getElementById('settings-bandwidth').value = cfg.max_bandwidth_mbps;
        if (cfg.max_disk_mb) document.getElementById('settings-disk').value = cfg.max_disk_mb;
      }

      App.downloads.load();
      App.dashboard.loadNetworkData();
      App.networkCode.load();
      App.dashboard._loading = false;
    },

    updateFull: function(data) {
      if (data.node_id) {
        var el = document.getElementById('node-id');
        var short = data.node_id.substring(0, 8);
        el.textContent = short;
        el.title = data.node_id;
        el.dataset.fullId = data.node_id;
        el.style.cursor = 'pointer';
        if (!el.dataset.bound) {
          el.dataset.bound = '1';
          el.addEventListener('click', function() {
            var fullId = el.dataset.fullId;
            var short = el.textContent;
            U.copyToClipboard(fullId, {
              btn: el,
              successLabel: I18n.t('nav.copied'),
              resetLabel: short,
              duration: 1200,
            });
          });
        }
      }
      if (data.version) document.getElementById('version').textContent = 'v' + data.version;
      if (data.uptime_seconds !== undefined) document.getElementById('uptime').textContent = U.formatUptime(data.uptime_seconds);
      if (data.tier) {
        U.setTierBadge('tier-badge', data.tier);
        U.setTierBadge('credit-tier', data.tier);
      }

      App.dashboard.updateStats(data);

      if (data.hardware) {
        var hw = data.hardware;
        S._gpuInference = !!hw.gpu_inference;
        var gpuEl = document.getElementById('node-gpu');
        var gpuBadge = document.getElementById('node-gpu-badge');
        if (hw.gpu_name) {
          gpuEl.textContent = hw.gpu_name;
          if (gpuBadge) {
            if (hw.gpu_inference) {
              var backendLabel = hw.inference_backend || 'GPU';
              gpuBadge.textContent = I18n.t('hw.gpu_mode_label', { backend: backendLabel });
              gpuBadge.className = 'node-mode-badge node-mode-gpu';
              gpuBadge.title = I18n.t('hw.gpu_mode_tip');
            } else {
              gpuBadge.textContent = I18n.t('hw.mode_cpu');
              gpuBadge.className = 'node-mode-badge node-mode-cpu';
              gpuBadge.title = I18n.t('hw.cpu_mode_tip');
            }
          }
          if (hw.gpu_vram_mb) {
            var vramUsed = hw.gpu_vram_used_mb || 0;
            var vramTotal = hw.gpu_vram_mb;
            var vramEl = document.getElementById('node-vram');

            var vramLabel = document.getElementById('vram-label');
            if (hw.gpu_inference) {
              if (vramLabel) vramLabel.textContent = I18n.t('hw.vram');
              // GPU mode: show model-estimated VRAM for loaded models
              var activeVramMb = 0;
              if (App.data.cache.models && App.data.cache.models.length) {
                App.data.cache.models.forEach(function(m) {
                  if (m.status === 'loaded' && m.estimated_vram_mb) activeVramMb += m.estimated_vram_mb;
                });
              }
              var displayUsed = activeVramMb > 0 ? activeVramMb : vramUsed;
              if (activeVramMb > 0 && vramUsed > activeVramMb + 200) {
                vramEl.textContent = I18n.t('hw.vram_active', { active: U.formatMB(activeVramMb), total: U.formatMB(vramTotal) });
                vramEl.title = I18n.t('hw.vram_reserved_tip', { used: U.formatMB(vramUsed) });
              } else {
                vramEl.textContent = U.formatMB(displayUsed) + ' / ' + U.formatMB(vramTotal);
                vramEl.title = '';
              }
              var vramPct = vramTotal > 0 ? (displayUsed / vramTotal * 100) : 0;
              document.getElementById('vram-bar').style.width = vramPct.toFixed(1) + '%';
              document.getElementById('vram-bar').className = vramPct > 90 ? 'fill red' : (vramPct > 70 ? 'fill orange' : 'fill cyan');
              App.dashboard._updateContribution(vramPct, 'vram');
            } else {
              if (vramLabel) vramLabel.textContent = I18n.t('hw.vram_idle');
              // CPU mode: contribution bar reflects RAM usage by loaded models,
              // not GPU VRAM (which is idle — driver baseline only).
              vramEl.textContent = U.formatMB(vramUsed) + ' / ' + U.formatMB(vramTotal);
              vramEl.title = I18n.t('hw.vram_idle_tip');
              var ramForModels = hw.process_rss_mb || 0;
              var ramPctForBar = hw.total_ram_mb > 0 ? (ramForModels / hw.total_ram_mb * 100) : 0;
              document.getElementById('vram-bar').style.width = ramPctForBar.toFixed(1) + '%';
              document.getElementById('vram-bar').className = ramPctForBar > 90 ? 'fill red' : (ramPctForBar > 70 ? 'fill orange' : 'fill cyan');
              App.dashboard._updateContribution(ramPctForBar, 'ram');
            }
          }
        } else {
          gpuEl.textContent = I18n.t('hw.none');
          if (gpuBadge) {
            gpuBadge.textContent = I18n.t('hw.mode_cpu_only');
            gpuBadge.className = 'node-mode-badge node-mode-cpu';
            gpuBadge.title = I18n.t('hw.cpu_only_tip');
          }
          document.getElementById('node-vram').textContent = '\u2014';
          document.getElementById('vram-bar').style.width = '0%';
        }
        document.getElementById('node-cpu').textContent = hw.cpu_name ? hw.cpu_name + ' ' + I18n.t('hw.cores', { cores: hw.cpu_cores }) : I18n.t('hw.unknown_cpu');

        if (hw.total_ram_mb) {
          document.getElementById('ram-total').textContent = '/ ' + U.formatMB(hw.total_ram_mb);
          // Show per-process RSS (this node's actual memory) rather than system-wide
          var processRss = hw.process_rss_mb || 0;
          var ramUsed = processRss > 0 ? processRss : (hw.used_ram_mb || 0);
          var ramEl = document.getElementById('ram-used');
          ramEl.textContent = U.formatMB(ramUsed);
          if (processRss > 0) {
            ramEl.title = U.formatMB(processRss) + '\n\n' +
              I18n.t(S._gpuInference ? 'hw.ram_tip_gpu' : 'hw.ram_tip_cpu') +
              '\n\n' + U.formatMB(hw.used_ram_mb || 0) + ' / ' + U.formatMB(hw.total_ram_mb);
          }
          var ramPct = hw.total_ram_mb > 0 ? (ramUsed / hw.total_ram_mb * 100) : 0;
          document.getElementById('ram-bar').style.width = ramPct.toFixed(1) + '%';
          document.getElementById('ram-bar').className = ramPct > 90 ? 'fill red' : (ramPct > 70 ? 'fill orange' : 'fill green');
        }
        if (hw.total_disk_mb) {
          document.getElementById('disk-total').textContent = '/ ' + U.formatMB(hw.total_disk_mb);
          var diskUsed = hw.used_disk_mb || 0;
          document.getElementById('disk-used').textContent = U.formatMB(diskUsed);
          var diskPct = hw.total_disk_mb > 0 ? (diskUsed / hw.total_disk_mb * 100) : 0;
          var diskBar = document.getElementById('disk-bar');
          diskBar.style.width = diskPct.toFixed(1) + '%';
          diskBar.className = diskPct > 90 ? 'fill red' : (diskPct > 70 ? 'fill orange' : 'fill accent');
        }
      }

      if (data.hosted_shards !== undefined) document.getElementById('hosted-shards').textContent = data.hosted_shards;
    },

    updateStats: function(data) {
      if (data.uptime_seconds !== undefined) {
        document.getElementById('uptime').textContent = U.formatUptime(data.uptime_seconds);
      }

      // Helper: track stat history & render sparkline + trend arrow
      function _trackStat(key, value, elId) {
        if (value === undefined) return;
        var hist = S.statHistory[key];
        if (!hist) { S.statHistory[key] = []; hist = S.statHistory[key]; }
        hist.push(value);
        if (hist.length > 20) hist.shift();

        // Update value display
        var valEl = document.getElementById(elId);
        if (valEl) valEl.textContent = typeof value === 'number' ? value.toLocaleString() : value;

        // Trend arrow
        var trendEl = document.getElementById(elId + '-trend');
        if (trendEl && hist.length >= 2) {
          var prev = hist[hist.length - 2], cur = hist[hist.length - 1];
          if (cur > prev) {
            trendEl.className = 'stat-trend trend-up';
            trendEl.textContent = '\u25B2';
          } else if (cur < prev) {
            trendEl.className = 'stat-trend trend-down';
            trendEl.textContent = '\u25BC';
          } else {
            trendEl.className = 'stat-trend trend-flat';
            trendEl.textContent = '\u2192';
          }
        }

        // Mini sparkline
        var sparkEl = document.getElementById(elId + '-spark');
        if (sparkEl && hist.length >= 2) {
          var min = Math.min.apply(null, hist);
          var max = Math.max.apply(null, hist);
          var range = max - min;
          var isFlat = range === 0;
          sparkEl.innerHTML = '';
          hist.forEach(function(v, i) {
            var bar = document.createElement('div');
            bar.className = 'spark-bar';
            // Flat data: show a centered 6px line; varied data: scale to 16px
            var h = isFlat ? 6 : Math.max(2, ((v - min) / range) * 16);
            bar.style.height = h + 'px';
            if (isFlat) bar.style.opacity = '0.25';
            sparkEl.appendChild(bar);
          });
        }
      }

      if (data.peers !== undefined) {
        _trackStat('peers', data.peers, 'stat-peers');
        var lanBadge = document.getElementById('lan-peer-badge');
        if (lanBadge) {
          if (data.lan_peers && data.lan_peers > 0) {
            lanBadge.textContent = data.lan_peers + ' ' + I18n.t('dashboard.lan_badge');
            lanBadge.style.display = 'inline-block';
          } else {
            lanBadge.style.display = 'none';
          }
        }
      }
      if (data.credits !== undefined) {
        var bal, earned, spent;
        if (typeof data.credits === 'object') {
          bal = data.credits.balance;
          earned = data.credits.lifetime_earned || 0;
          spent = data.credits.lifetime_spent || 0;
        } else {
          bal = data.credits;
          earned = 0;
          spent = 0;
        }
        _trackStat('credits', bal, 'stat-credits');
        document.getElementById('credit-balance').textContent = bal.toLocaleString();
        document.getElementById('credit-earned').textContent = '+' + earned.toLocaleString();
        document.getElementById('credit-spent').textContent = '-' + spent.toLocaleString();
        var prevBal = S.creditHistory.length > 0 ? S.creditHistory[S.creditHistory.length - 1]._bal : bal;
        var delta = bal - prevBal;
        S.creditHistory.push({ _bal: bal, v: delta });
        if (S.creditHistory.length > 30) S.creditHistory.shift();
        U.renderSparkline('credit-sparkline', S.creditHistory.map(function(e) { return e.v; }));
      }
      if (data.requests_served !== undefined) _trackStat('served', data.requests_served, 'stat-served');
      if (data.requests_made !== undefined) _trackStat('requests', data.requests_made, 'stat-requests-made');
      if (data.forwards_served !== undefined) _trackStat('forwards', data.forwards_served, 'stat-forwards');
      if (data.active_requests !== undefined) _trackStat('active', data.active_requests, 'stat-active');

      App.modeIndicator.update(data, S._cachedProviderData);

      if (typeof NeuralBg !== 'undefined') NeuralBg.updateState(data);
    },

    renderModels: function(models, cloudModels) {
      // models cached in App.data.cache.models
      var list = document.getElementById('models-list');
      var empty = document.getElementById('models-empty');
      var loading = document.getElementById('models-loading');
      if (loading) loading.remove();

      // Split cloud models into API-key providers vs subscription providers
      var apiModels = [];
      var subscriptionModels = [];
      if (cloudModels && cloudModels.length > 0) {
        cloudModels.forEach(function(cm) {
          if (typeof isSubscriptionProvider === 'function' && isSubscriptionProvider(cm.provider)) {
            subscriptionModels.push(cm);
          } else {
            apiModels.push(cm);
          }
        });
      }
      var hasCloud = apiModels.length > 0;
      var hasSubscription = subscriptionModels.length > 0;

      if ((!models || models.length === 0) && !hasCloud && !hasSubscription) {
        list.innerHTML = '';
        empty.style.display = '';
        var _sb = document.getElementById('models-stats-bar');
        if (_sb) _sb.style.display = 'none';
        return;
      }

      // Filter out ghost models
      models = models.filter(function(m) {
        if (m.local || m.hosted_shards > 0) return true;
        if (m.peers_hosting > 0) return true;
        if (m.acquisition === 'downloading') return true;
        var anyHolder = (m.shards || []).some(function(s) { return s.holders > 0; });
        return anyHolder;
      });

      if (models.length === 0 && !hasCloud && !hasSubscription) {
        list.innerHTML = '';
        empty.style.display = '';
        var _sb2 = document.getElementById('models-stats-bar');
        if (_sb2) _sb2.style.display = 'none';
        return;
      }

      empty.style.display = 'none';
      list.innerHTML = '';

      // Quick stats
      var statsBar = document.getElementById('models-stats-bar');
      if (statsBar) {
        var statLocal = models.filter(function(m) { return m.local || m.hosted_shards > 0; }).length;
        var statReady = models.filter(function(m) {
          var hc = m.hosted_shards || 0, sc = m.shard_count || (m.shards || []).length;
          return m.status === 'loaded' || m.status === 'ready' || (hc === sc && sc > 0);
        }).length;
        var statNet = models.filter(function(m) { return !m.local && !(m.hosted_shards > 0) && m.peers_hosting > 0; }).length;
        var statCloudTotal = hasCloud ? apiModels.length : 0;
        var statProviders = 0;
        if (hasCloud) {
          var _pset = {};
          apiModels.forEach(function(cm) { _pset[cm.provider || 'cloud'] = 1; });
          statProviders = Object.keys(_pset).length;
        }
        document.getElementById('stat-chip-ready-val').textContent = statReady;
        document.getElementById('stat-chip-network-val').textContent = statNet;
        document.getElementById('stat-chip-cloud-val').textContent = statCloudTotal;
        document.getElementById('stat-chip-providers-val').textContent = statProviders;
        statsBar.style.display = '';
        var netChip = document.getElementById('stat-chip-network');
        if (netChip) netChip.style.display = statNet > 0 ? '' : 'none';
        var cloudGroup = document.getElementById('stat-group-cloud');
        var sep = statsBar.querySelector('.models-stat-sep');
        if (cloudGroup) cloudGroup.style.display = hasCloud ? '' : 'none';
        if (sep) sep.style.display = hasCloud ? '' : 'none';
        // Subscription stats chip
        var subGroup = document.getElementById('stat-group-subscription');
        var subSep = document.getElementById('models-stat-sep-sub');
        if (subGroup) subGroup.style.display = hasSubscription ? '' : 'none';
        if (subSep) subSep.style.display = hasSubscription ? '' : 'none';
        var subValEl = document.getElementById('stat-chip-subscription-val');
        if (subValEl && hasSubscription) subValEl.textContent = subscriptionModels.length;
      }

      // Sort swarm models
      var swarmSort = S._swarmModelSort || 'problems';
      function _modelProblemScore(m) {
        var shards = m.shards || [];
        var shardCount = m.shard_count || shards.length || 0;
        var hostedShards = m.hosted_shards || 0;
        var globalAvail = m.global_available || hostedShards;
        var isReady = m.status === 'loaded' || m.status === 'ready' || (globalAvail === shardCount && shardCount > 0);
        var isDownloading = m.acquisition === 'downloading';
        // Lower = more urgent (sort ascending)
        if (m.status === 'loaded') return 10; // active — show near top
        if (isDownloading && !isReady) return 20; // downloading, not ready
        var fragile = shards.filter(function(s) { return (s.holders || 0) === 1; }).length;
        var missing = shards.filter(function(s) { return !s.local && (s.holders || 0) === 0; }).length;
        if (missing > 0) return 30; // incomplete
        if (fragile > 0) return 40; // fragile
        if (isReady) return 80; // healthy
        return 60;
      }
      function _sortModels(arr, mode) {
        var sorted = arr.slice();
        if (mode === 'problems') {
          sorted.sort(function(a, b) {
            var sa = _modelProblemScore(a), sb = _modelProblemScore(b);
            if (sa !== sb) return sa - sb;
            var na = (a.name || a.id).toLowerCase(), nb = (b.name || b.id).toLowerCase();
            return na < nb ? -1 : na > nb ? 1 : 0;
          });
        } else if (mode === 'az') {
          sorted.sort(function(a, b) {
            var na = (a.name || a.id).toLowerCase(), nb = (b.name || b.id).toLowerCase();
            return na < nb ? -1 : na > nb ? 1 : 0;
          });
        } else if (mode === 'za') {
          sorted.sort(function(a, b) {
            var na = (a.name || a.id).toLowerCase(), nb = (b.name || b.id).toLowerCase();
            return na > nb ? -1 : na < nb ? 1 : 0;
          });
        } else if (mode === 'status') {
          var rank = { loaded: 0, ready: 1, downloading: 2, partial: 3, available: 4, network: 5 };
          sorted.sort(function(a, b) {
            var ra = rank[a.status] !== undefined ? rank[a.status] : 9;
            var rb = rank[b.status] !== undefined ? rank[b.status] : 9;
            if (ra !== rb) return ra - rb;
            var na = (a.name || a.id).toLowerCase(), nb = (b.name || b.id).toLowerCase();
            return na < nb ? -1 : na > nb ? 1 : 0;
          });
        } else if (mode === 'size') {
          sorted.sort(function(a, b) { return (b.total_size_bytes || 0) - (a.total_size_bytes || 0); });
        } else if (mode === 'shards') {
          sorted.sort(function(a, b) { return (b.hosted_shards || 0) - (a.hosted_shards || 0); });
        }
        return sorted;
      }
      models = _sortModels(models, swarmSort);

      // Swarm models section
      var swarmBody;
      if (models.length > 0) {
        var swarmSection = document.createElement('details');
        swarmSection.className = 'models-section';
        swarmSection.open = true;
        var swarmReadyCount = models.filter(function(m) {
          var hc = m.hosted_shards || 0, sc = m.shard_count || (m.shards || []).length;
          return m.status === 'loaded' || m.status === 'ready' || (hc === sc && sc > 0);
        }).length;
        var swarmMeta = I18n.t('dashboard.models_count', { count: models.length, ready: swarmReadyCount });
        swarmSection.innerHTML = '<summary class="models-section-header">' +
          '<img src="/static/icons/swarm.svg" width="16" height="16" alt="" aria-hidden="true" class="models-section-logo">' +
          '<span class="models-section-title">' + U.escapeHtml(I18n.t('dashboard.swarm_models')) + '</span>' +
          '<span class="models-section-count">' + swarmMeta + '</span>' +
          '<select class="swarm-model-sort" id="swarm-model-sort" title="' + U.escapeHtml(I18n.t('dashboard.sort_title')) + '">' +
            '<option value="problems"' + (swarmSort === 'problems' ? ' selected' : '') + '>' + U.escapeHtml(I18n.t('dashboard.sort_problems')) + '</option>' +
            '<option value="az"' + (swarmSort === 'az' ? ' selected' : '') + '>' + U.escapeHtml(I18n.t('dashboard.sort_az')) + '</option>' +
            '<option value="za"' + (swarmSort === 'za' ? ' selected' : '') + '>' + U.escapeHtml(I18n.t('dashboard.sort_za')) + '</option>' +
            '<option value="status"' + (swarmSort === 'status' ? ' selected' : '') + '>' + U.escapeHtml(I18n.t('dashboard.sort_status')) + '</option>' +
            '<option value="size"' + (swarmSort === 'size' ? ' selected' : '') + '>' + U.escapeHtml(I18n.t('dashboard.sort_size')) + '</option>' +
            '<option value="shards"' + (swarmSort === 'shards' ? ' selected' : '') + '>' + U.escapeHtml(I18n.t('dashboard.sort_local_shards')) + '</option>' +
          '</select>' +
          '</summary>';
        swarmBody = document.createElement('div');
        swarmBody.className = 'models-section-body';
        swarmSection.appendChild(swarmBody);
        list.appendChild(swarmSection);

        // Wire sort change handler
        var sortEl = document.getElementById('swarm-model-sort');
        if (sortEl) {
          sortEl.addEventListener('change', function(e) {
            e.stopPropagation(); // Don't toggle the <details>
            S._swarmModelSort = this.value;
            try { localStorage.setItem(App.MODEL_SORT_KEY, this.value); } catch(e2) {}
            App.models.load();
          });
          sortEl.addEventListener('click', function(e) { e.stopPropagation(); });
        }
      }

      models.forEach(function(m) {
        var shards = m.shards || [];
        var shardCount = m.shard_count || shards.length || 0;
        var hostedShards = m.hosted_shards || 0;
        var globalAvail = m.global_available || hostedShards;
        var isDownloading = m.acquisition === 'downloading';
        var isReady = m.status === 'loaded' || m.status === 'ready' || (globalAvail === shardCount && shardCount > 0);
        // Auto-manage may download local copies of a model that is already READY via peers.
        // In that case, show as Ready (not Downloading) — the download is just local caching.
        var isCachingLocally = isDownloading && isReady;
        var isPartial = !isReady && hostedShards > 0 && hostedShards < shardCount;
        var safeId = U.safeId(m.id || '');

        var card = document.createElement('div');
        var isCompact = !S._expandedModels[m.id];
        // Any shard with zero network replicas → the model can't run anywhere.
        var unusable = shards.some(function(s) { return !s.local && (s.holders || 0) === 0; });
        card.className = 'model-card' + (isReady ? ' ready' : (isDownloading ? ' downloading' : (isPartial ? ' partial' : ''))) + (isCompact ? ' compact' : '') + (unusable ? ' cb-unusable' : '');
        card.setAttribute('data-model-id', m.id);

        // --- Composite health badge (single badge replacing 4 separate indicators) ---
        var compositeBadgeClass, compositeBadgeLabel, compositeBadgeTitle;
        var fragileCount = shards.filter(function(s) { return (s.holders || 0) === 1; }).length;
        var networkMissingCount = shards.filter(function(s) { return !s.local && (s.holders || 0) === 0; }).length;
        if (m.status === 'loaded') {
          compositeBadgeClass = 'cb-active';
          compositeBadgeLabel = I18n.t('dashboard.status_active');
          compositeBadgeTitle = I18n.t('dashboard.cb_active_tip');
        } else if (isReady && !isDownloading) {
          compositeBadgeClass = 'cb-ready';
          compositeBadgeLabel = I18n.t('dashboard.status_ready');
          compositeBadgeTitle = I18n.t('dashboard.cb_ready_tip');
        } else if (isCachingLocally) {
          compositeBadgeClass = 'cb-ready';
          compositeBadgeLabel = I18n.t('dashboard.status_ready');
          compositeBadgeTitle = I18n.t('dashboard.cb_caching_tip');
        } else if (isDownloading) {
          compositeBadgeClass = 'cb-downloading';
          compositeBadgeLabel = I18n.t('dashboard.status_downloading');
          compositeBadgeTitle = I18n.t('dashboard.cb_downloading_tip');
        } else if (networkMissingCount > 0) {
          compositeBadgeClass = 'cb-incomplete';
          compositeBadgeLabel = I18n.t('dashboard.cb_incomplete', { count: networkMissingCount });
          compositeBadgeTitle = I18n.t('dashboard.cb_incomplete_tip', { count: networkMissingCount, total: shardCount });
        } else if (fragileCount > 0) {
          compositeBadgeClass = 'cb-fragile';
          compositeBadgeLabel = I18n.t('dashboard.cb_fragile', { count: fragileCount });
          compositeBadgeTitle = I18n.t('dashboard.cb_fragile_tip', { count: fragileCount });
        } else if (isPartial) {
          compositeBadgeClass = 'cb-incomplete';
          compositeBadgeLabel = I18n.t('dashboard.local_status', { hosted: hostedShards, total: shardCount });
          compositeBadgeTitle = I18n.t('dashboard.cb_partial_tip');
        } else {
          compositeBadgeClass = 'cb-network';
          compositeBadgeLabel = I18n.t('dashboard.status_on_network');
          compositeBadgeTitle = I18n.t('dashboard.cb_network_tip');
        }
        var compositeBadgeHtml = '<span class="composite-badge ' + compositeBadgeClass + '" title="' + U.escapeHtml(compositeBadgeTitle) + '">' +
          '<span class="cb-dot"></span>' + U.escapeHtml(compositeBadgeLabel) + '</span>';

        // --- Availability bar (compact shard strip) ---
        var availBarHtml = '';
        if (shards.length > 0) {
          availBarHtml = '<div class="availability-bar" data-avail-bar="' + safeId + '" title="' + U.escapeHtml(I18n.t('dashboard.avail_bar_tip', { local: hostedShards, total: shardCount })) + '">';
          shards.forEach(function(s) {
            var segClass = 'seg-missing';
            if (s.local && s.in_vram) segClass = 'seg-active';
            else if (s.local) segClass = 'seg-nominal';
            else if (s.download && (s.download.state === 'Downloading' || s.download.state === 'Verifying')) segClass = 'seg-downloading';
            else if (s.peer_downloads && s.peer_downloads.length > 0) segClass = 'seg-downloading';
            else if (s.holders > 0) {
              segClass = (s.holders || 0) === 1 ? 'seg-warning' : 'seg-peer';
            }
            else segClass = 'seg-problem';
            // Missing but no holders at all = problem
            if (!s.local && (s.holders || 0) === 0 && !s.download) segClass = shardCount > 1 ? 'seg-problem' : 'seg-missing';
            availBarHtml += '<div class="avail-seg ' + segClass + '"></div>';
          });
          availBarHtml += '</div>';
        }

        // --- Detail badges (shown only in expanded mode) ---
        var detailBadgesHtml = '';
        var detailParts = [];
        // Trust badge
        if (m.trust_level === 'network_popular') {
          detailParts.push('<span class="badge-trust badge-trust-popular" title="' + U.escapeHtml(I18n.t('dashboard.trust_popular')) + '">' + U.escapeHtml(I18n.t('dashboard.badge_popular')) + '</span>');
        } else if (m.trust_level === 'demand_verified') {
          detailParts.push('<span class="badge-trust badge-trust-verified" title="' + U.escapeHtml(I18n.t('dashboard.trust_verified')) + '">' + U.escapeHtml(I18n.t('dashboard.badge_verified')) + '</span>');
        } else if (m.trust_level === 'pinned') {
          detailParts.push('<span class="badge-trust badge-trust-pinned" title="' + U.escapeHtml(I18n.t('dashboard.trust_pinned')) + '">' + U.escapeHtml(I18n.t('dashboard.badge_pinned')) + '</span>');
        }
        // (Encrypted pipeline badge is rendered as an integrated toggle inside the
        //  .mce-pipeline chip below — no standalone floating lock icon anymore.)
        // Source label
        if (m.source === 'network' && hostedShards === 0) {
          detailParts.push('<span class="badge badge-orange" title="' + U.escapeHtml(I18n.t('dashboard.badge_remote')) + '">' + U.escapeHtml(I18n.t('dashboard.badge_remote_label')) + '</span>');
        }
        if (detailParts.length > 0) {
          detailBadgesHtml = '<div class="model-card-detail-badges">' + detailParts.join('') + '</div>';
        }

        // Gear + info buttons
        var gearHtml = '<button class="model-gear-btn" data-am-gear="' + U.escapeHtml(m.id) + '" title="' + U.escapeHtml(I18n.t('dashboard.gear_title')) + '">&#9881;</button>';
        var metaBtnHtml = m.has_header ? '<button class="model-meta-btn" data-meta-toggle="' + U.escapeHtml(m.id) + '" title="' + U.escapeHtml(I18n.t('models.metadata_header')) + '">&#9432;</button>' : '';

        // Swarm health summary badge — shown in the left column of the expanded card.
        // Derived from per-shard holder counts across the network.
        var healthBadgeHtml = '';
        if (shards.length > 0) {
          var totalShards = shards.length;
          var totalHolders = 0;
          var wellReplicated = 0, adequate = 0, fragile = 0, networkMissing = 0;
          shards.forEach(function(s) {
            var holders = s.holders || 0;
            totalHolders += holders;
            if (holders >= 3) wellReplicated++;
            else if (holders === 2) adequate++;
            else if (holders === 1) fragile++;
            else networkMissing++;
          });
          var avgHolders = totalShards > 0 ? (totalHolders / totalShards) : 0;
          var healthLabel, healthClass;
          if (networkMissing === totalShards) { healthLabel = I18n.t('dashboard.health_unavailable_label'); healthClass = 'health-low'; }
          else if (networkMissing > 0) { healthLabel = I18n.t('dashboard.health_incomplete'); healthClass = 'health-low'; }
          else if (fragile > 0) { healthLabel = I18n.t('dashboard.health_fragile'); healthClass = 'health-partial'; }
          else if (avgHolders >= 2) { healthLabel = I18n.t('dashboard.health_healthy'); healthClass = 'health-full'; }
          else { healthLabel = I18n.t('dashboard.health_good'); healthClass = 'health-good'; }
          var healthDetail = '';
          if (healthClass === 'health-full') healthDetail = I18n.t('dashboard.health_replicated', { avg: avgHolders.toFixed(1) });
          else if (healthClass === 'health-good') healthDetail = I18n.t('dashboard.health_distributed', { count: totalShards });
          else if (fragile > 0) healthDetail = I18n.t('dashboard.health_under_replicated', { count: fragile });
          else if (networkMissing === totalShards) healthDetail = I18n.t('dashboard.health_no_shards_available');
          else if (networkMissing > 0) healthDetail = I18n.t('dashboard.health_missing', { count: networkMissing });
          healthBadgeHtml = '<div class="mce-health ' + healthClass + '">' +
            '<span class="mce-health-label">' + U.escapeHtml(healthLabel) + '</span>' +
            '<span class="mce-health-detail">' + U.escapeHtml(healthDetail) + '</span>' +
            '</div>';
        }

        // Pipeline encryption status — SwarmLLM requires the user to locally hold
        // BOTH the first and last shard to fully encrypt the inference pipeline.
        // Chip surfaces whether the guarantee is currently met, merely available
        // (both endpoints local), or unprotected (one/both endpoints missing).
        // Pipeline encryption — rendered as a standard .mce-section with a
        // state modifier (green/blue/amber) so it shares the panel language
        // with STATUS and CONFIG. Computes the data once, section markup is
        // assembled later alongside the other sections.
        var privacySectionHtml = '';
        if (shardCount > 1) {
          var hasFirst = !!m.has_first_shard;
          var hasLast  = !!m.has_last_shard;
          var encActive = !!m.encrypted_pipeline;
          var canToggle = hasFirst && hasLast;
          var encState; // { stateMod, badgeCls, icon, label, detail, tip, action }
          if (encActive) {
            encState = {
              stateMod: 'mce-section-state-green', badgeCls: 'cb-active',
              icon: '\uD83D\uDD12', label: I18n.t('enc.active'),
              detail: I18n.t('enc.active_detail'), tip: I18n.t('enc.active_tip'),
              action: I18n.t('enc.disable')
            };
          } else if (canToggle) {
            encState = {
              stateMod: 'mce-section-state-blue', badgeCls: 'cb-downloading',
              icon: '\uD83D\uDD0F', label: I18n.t('enc.available'),
              detail: I18n.t('enc.ready_detail'), tip: I18n.t('enc.ready_tip'),
              action: I18n.t('enc.enable_privacy')
            };
          } else {
            var missingParts2 = [];
            if (!hasFirst) missingParts2.push(I18n.t('dashboard.enc_missing_first'));
            if (!hasLast)  missingParts2.push(I18n.t('dashboard.enc_missing_last', { n: shardCount - 1 }));
            var missingText = missingParts2.length === 2
              ? I18n.t('enc.missing_both')
              : (missingParts2.length === 1 ? I18n.t('enc.missing_the', { which: missingParts2[0] }) : '');
            encState = {
              stateMod: 'mce-section-state-amber', badgeCls: 'cb-fragile',
              icon: '\uD83D\uDD13', label: I18n.t('enc.unavailable'),
              detail: I18n.t('enc.unprotected_detail', { missing: missingText }),
              tip: I18n.t('enc.unprotected_tip'),
              action: ''
            };
          }
          var toggleAttrs = canToggle
            ? ' data-enc-toggle="' + U.escapeHtml(m.id) + '" data-enc-ready="1" role="switch" aria-checked="' + (encActive ? 'true' : 'false') + '"'
            : '';
          var toggleCls = canToggle ? ' mce-section-toggleable' : '';
          var actionHtml2 = canToggle && encState.action
            ? '<span class="mce-section-action">' + U.escapeHtml(encState.action) + '</span>'
            : '';
          privacySectionHtml =
            '<div class="mce-section mce-section-privacy ' + encState.stateMod + toggleCls + '"' + toggleAttrs + ' title="' + U.escapeHtml(encState.tip) + '">' +
              '<div class="mce-section-header">' +
                '<div class="mce-section-title">' + U.escapeHtml(I18n.t('dashboard.section_privacy')) + '</div>' +
                actionHtml2 +
              '</div>' +
              '<div class="mce-section-body">' +
                '<span class="composite-badge ' + encState.badgeCls + '">' +
                  '<span class="mce-section-icon">' + encState.icon + '</span>' +
                  U.escapeHtml(encState.label) +
                '</span>' +
                '<div class="mce-section-detail">' + U.escapeHtml(encState.detail) + '</div>' +
              '</div>' +
            '</div>';
        }

        // Download progress bar
        var progressHtml = '';
        if (isDownloading && m.acquisition_progress) {
          var ap = m.acquisition_progress;
          var dlBytes = ap.downloaded_bytes || 0;
          var totalBytes = ap.total_bytes || 0;
          if (dlBytes > totalBytes && totalBytes > 0) dlBytes = totalBytes;
          var pct = totalBytes > 0 ? Math.min(100, Math.round((dlBytes / totalBytes) * 100)) : 0;
          var speed = ap.speed_bytes_per_sec || 0;
          var etaStr = '';
          if (speed > 0 && totalBytes > dlBytes) {
            etaStr = U.formatEta((totalBytes - dlBytes) / speed);
          }
          var dlShards2 = shards.filter(function(s) { return s.download || s.local; });
          var segmentCount = Math.max(dlShards2.length, shardCount);
          var segmentsHtml = '';
          if (segmentCount > 0) {
            var segW = (100 / segmentCount);
            for (var si = 0; si < segmentCount; si++) {
              var sh = shards.find(function(s) { return s.index === si; });
              var segPct = 0;
              if (sh && sh.local) segPct = 100;
              else if (sh && sh.download) segPct = sh.download.progress_pct || 0;
              segmentsHtml += '<div class="dl-seg" style="width:' + segW.toFixed(2) + '%;"><div class="dl-seg-fill" style="width:' + segPct + '%"></div></div>';
            }
          }
          var shardLabel;
          var localNow = shards.filter(function(s) { return s.local; }).length;
          var dlSource = ap.source || '';
          var dlTrigger = ap.trigger || '';
          var triggerText = dlTrigger === 'auto_manage' ? I18n.t('dashboard.auto_manage') : (dlTrigger === 'user' ? I18n.t('dashboard.manual') : '');
          var sourceText = dlSource === 'huggingface' ? I18n.t('dashboard.from_hf') : (dlSource === 'peers' ? I18n.t('dashboard.from_peers') : '');
          if (isCachingLocally) {
            shardLabel = I18n.t('dashboard.caching_label', { trigger: triggerText || I18n.t('dashboard.auto_manage'), local: localNow, total: shardCount });
          } else {
            // Show which specific shard is downloading (from shard_details)
            var dlShardIdx = '';
            if (ap.shard_details) {
              var activeShard = ap.shard_details.find(function(sd) { return sd.state === 'downloading'; });
              if (activeShard) dlShardIdx = I18n.t('dashboard.downloading_part', { n: activeShard.index + 1 });
            }
            shardLabel = (triggerText ? triggerText + ': ' : '') + I18n.t('dashboard.downloading_label') + dlShardIdx + (sourceText ? ' ' + sourceText : '') + I18n.t('dashboard.downloading_local', { local: localNow, total: shardCount });
          }
          var rightText = U.formatDlProgress(dlBytes, totalBytes, pct);
          if (speed > 0) rightText += ' \u00b7 ' + U.formatSpeed(speed);
          if (etaStr) rightText += I18n.t('dashboard.eta', { eta: etaStr });
          progressHtml = _buildProgressBar({ safeId: safeId, pct: pct, label: shardLabel, rightText: rightText, barContent: segmentsHtml });
        }

        // Per-shard download bars
        var perShardDlHtml = '';
        if (isDownloading && shards.length > 0 && shardCount <= 20) {
          var dlShardBars = shards.filter(function(s) {
            return s.download && s.download.state === 'Downloading';
          });
          if (dlShardBars.length > 1) {
            perShardDlHtml = '<div class="per-shard-dl">';
            dlShardBars.forEach(function(s) {
              var pct2 = s.download.progress_pct || 0;
              var bytes = s.download.downloaded_bytes || 0;
              var total = s.download.total_bytes || s.size_bytes || 0;
              perShardDlHtml += '<div class="per-shard-dl-row">' +
                '<span class="per-shard-dl-label">' + U.escapeHtml(I18n.t('shard.part_n', { n: s.index + 1 })) + '</span>' +
                '<div class="per-shard-dl-bar"><div class="per-shard-dl-fill" style="width:' + pct2 + '%"></div></div>' +
                '<span class="per-shard-dl-pct">' + U.formatBytes(bytes) + '/' + U.formatBytes(total) + ' (' + pct2 + '%)</span>' +
                '</div>';
            });
            perShardDlHtml += '</div>';
          }
        }

        // --- Parse architecture + quantization from model ID ---
        var modelId = m.id || '';
        var archKey = modelIconKey(modelId);
        var archTag = archKey ? '<span class="model-tag tag-arch">' + U.escapeHtml(archKey) + '</span>' : '';
        var quantMatch = modelId.match(/[._-](q[0-9]+[_-]?k?[_-]?[a-z]*)/i);
        var quantTag = quantMatch ? '<span class="model-tag tag-quant">' + U.escapeHtml(quantMatch[1].toUpperCase().replace(/-/g, '_')) + '</span>' : '';

        // --- Config rows (key/value pairs for the CONFIG section) ---
        var configRows = [];
        if (archKey)    configRows.push(['dashboard.info_arch',  '<span class="mce-info-pill">' + U.escapeHtml(archKey) + '</span>']);
        if (quantMatch) configRows.push(['dashboard.info_quant', '<span class="mce-info-pill">' + U.escapeHtml(quantMatch[1].toUpperCase().replace(/-/g, '_')) + '</span>']);
        configRows.push(['dashboard.info_size', U.formatBytes(m.total_size_bytes || 0)]);
        if (shardCount > 0) {
          configRows.push(['dashboard.info_shards', String(shardCount)]);
        }
        // Mode (CPU/GPU) — single word
        configRows.push(['dashboard.info_mode', S._gpuInference ? I18n.t('dashboard.mode_gpu') : I18n.t('dashboard.mode_cpu')]);
        // VRAM fit — only when GPU mode; in CPU mode the Mode row already conveys this.
        if (m.estimated_vram_mb && S._gpuInference) {
          var totalVram = (App.data.cache.stats && App.data.cache.stats.hardware && App.data.cache.stats.hardware.gpu_vram_mb) || 0;
          var fitClass = 'fit-no', fitLabel = U.formatMB(m.estimated_vram_mb);
          if (totalVram > 0) {
            var ratio = m.estimated_vram_mb / totalVram;
            if (ratio <= 0.85) { fitClass = 'fit-yes'; fitLabel = '\u2713 ' + fitLabel; }
            else if (ratio <= 1.05) { fitClass = 'fit-tight'; fitLabel = '\u2248 ' + fitLabel; }
            else { fitClass = 'fit-no'; fitLabel = '\u2717 ' + fitLabel; }
          }
          configRows.push(['dashboard.info_vram', '<span class="vram-fit ' + fitClass + '" title="' + U.escapeHtml(I18n.t('dashboard.vram_fit_tip', { est: U.formatMB(m.estimated_vram_mb), total: totalVram > 0 ? U.formatMB(totalVram) : '?' })) + '">' + fitLabel + '</span>']);
        }
        // Trust is rendered in the CONFIG section header (top-right), not as
        // a grid row — frees a cell and surfaces trust next to "Config".
        var trustHeaderHtml = (detailBadgesHtml && m.trust_level) ? detailBadgesHtml : '';
        var configGridHtml = configRows.map(function(row) {
          return '<dt>' + U.escapeHtml(I18n.t(row[0])) + '</dt><dd>' + row[1] + '</dd>';
        }).join('');

        // Peer count line for STATUS section
        var peerLineHtml = '';
        if (m.peers_hosting > 0) {
          peerLineHtml = '<div class="mce-status-peers"><span class="mce-status-icon">\u2B65</span>' + U.escapeHtml(I18n.t('dashboard.peer_count', { count: m.peers_hosting })) + '</div>';
        } else if (hostedShards > 0) {
          peerLineHtml = '<div class="mce-status-peers mce-warn" title="' + U.escapeHtml(I18n.t('dashboard.local_only_tip')) + '"><span class="mce-status-icon">\u26A0</span>' + U.escapeHtml(I18n.t('dashboard.local_only')) + '</div>';
        }

        // Missing files warning
        var fileIndicators = '';
        if (hostedShards > 0 || isDownloading) {
          var hasManifest = m.has_manifest !== false;
          var hasHeader = m.has_header !== false;
          if (!hasManifest || !hasHeader) {
            var missingFiles = [];
            if (!hasManifest) missingFiles.push(I18n.t('dashboard.missing_manifest'));
            if (!hasHeader) missingFiles.push(I18n.t('dashboard.missing_header'));
            fileIndicators = '<span class="meta-sep">\u00B7</span><span class="meta-item meta-warn" title="' + U.escapeHtml(I18n.t('dashboard.missing_files', { files: missingFiles.join(', ') })) + '">\u26A0 ' + I18n.t('dashboard.missing_warning', { files: missingFiles.join(' + ') }) + '</span>';
          }
        }

        // --- Styled action buttons ---
        var actionHtml = '';
        if (m.status === 'loaded') {
          actionHtml = '<button class="btn-action" data-unload-model="' + U.escapeHtml(m.id) + '" title="' + U.escapeHtml(I18n.t('dashboard.unload_tip')) + '">' + U.escapeHtml(I18n.t('dashboard.btn_unload_all')) + '</button>';
        } else if (isReady) {
          actionHtml = '<button class="btn-action btn-primary-action" data-select-model="' + U.escapeHtml(m.id) + '">' + U.escapeHtml(I18n.t('dashboard.btn_use')) + '</button>';
        } else if (isDownloading) {
          actionHtml = '<button class="btn-action btn-danger" data-cancel-download="' + U.escapeHtml(m.id) + '" title="' + U.escapeHtml(I18n.t('shard.cancel_download')) + '">&times; ' + U.escapeHtml(I18n.t('actions.cancel')) + '</button>';
        } else if (m.source === 'network' || m.status === 'available' || m.status === 'partial') {
          actionHtml = '<button class="btn-action btn-download" data-request-model="' + U.escapeHtml(m.id) + '">' + U.escapeHtml(I18n.t('models.download')) + '</button>';
        }

        var removeHtml = '';
        if (hostedShards > 0 && !isDownloading) {
          removeHtml = '<button class="btn-action btn-danger" data-remove-model="' + U.escapeHtml(m.id) + '">' + U.escapeHtml(I18n.t('dashboard.btn_remove_model')) + '</button>';
        }

        var name = U.formatModelDisplayName(m.name || m.id);
        var creatorIconHtml = providerIconHtml(modelIconKey(m.id), 20);
        var chevronHtml = '<span class="model-expand-chevron" title="' + U.escapeHtml(I18n.t('dashboard.expand_collapse')) + '">&#9662;</span>';

        // Active loaded class for pulsing border
        if (m.status === 'loaded') card.classList.add('active-loaded');

        // Card HTML — compact by default with availability bar, expand for full shard grid
        card.innerHTML =
          '<div class="model-card-title">' +
            '<div class="model-card-title-main">' +
              '<div class="model-card-name-row">' +
                creatorIconHtml +
                '<span class="model-name" title="' + U.escapeHtml(m.id) + '">' + U.escapeHtml(name) + '</span>' +
                compositeBadgeHtml +
              '</div>' +
            '</div>' +
            // Coverage ribbon sits in the title row so it aligns horizontally
            // with the shard list / matrix columns below. Shown only in expanded
            // mode (compact mode uses the separate full-width availability bar).
            '<div class="model-card-title-health">' +
              (shards.length > 0 ? _buildCoverageRibbon(m, shards, safeId) : '') +
            '</div>' +
            '<div class="model-card-controls">' +
              metaBtnHtml + gearHtml + chevronHtml +
            '</div>' +
          '</div>' +
          '<div class="model-card-shards">' +
            progressHtml + perShardDlHtml +
            '<div class="model-card-expanded' + (m.encrypted_pipeline ? ' pipeline-encrypted' : '') + '">' +
              '<div class="mce-left">' +
                // STATUS — title + status badge inline; peer count on the right.
                // Health badge (fragile/degraded/etc.) drops into the body row.
                '<div class="mce-section mce-section-status">' +
                  '<div class="mce-section-header">' +
                    '<div class="mce-section-title-row">' +
                      '<div class="mce-section-title">' + U.escapeHtml(I18n.t('dashboard.section_status')) + '</div>' +
                      compositeBadgeHtml +
                    '</div>' +
                    peerLineHtml +
                  '</div>' +
                  (healthBadgeHtml ? '<div class="mce-section-body">' + healthBadgeHtml + '</div>' : '') +
                '</div>' +
                // PRIVACY — pipeline encryption (skipped for single-shard models).
                // Above CONFIG so the connector line lands higher and closer
                // to the endpoint shard rows on the right.
                privacySectionHtml +
                // CONFIG — static spec sheet: arch, quant, size, shards, mode, vram.
                // Trust badge sits in the header top-right.
                '<div class="mce-section mce-section-config">' +
                  '<div class="mce-section-header">' +
                    '<div class="mce-section-title">' + U.escapeHtml(I18n.t('dashboard.section_config')) + '</div>' +
                    trustHeaderHtml +
                  '</div>' +
                  '<dl class="mce-config-grid">' + configGridHtml + '</dl>' +
                '</div>' +
                '<div class="mce-actions">' + actionHtml + removeHtml + '</div>' +
                (fileIndicators ? '<div class="mce-file-warn">' + fileIndicators + '</div>' : '') +
              '</div>' +
              '<div class="mce-right" data-shard-detail="' + safeId + '">' +
                '<div class="mce-right-head">' +
                  _buildShardViewToggle() +
                '</div>' +
                '<div class="mce-right-body">' + _buildShardDetailBody(m, shards, safeId) + '</div>' +
                // Activity/Network ticker lives under the matrix — fills right-column dead space
                '<div class="model-ticker model-ticker-embedded" data-model-ticker="' + safeId + '" style="display:none"></div>' +
              '</div>' +
            '</div>' +
          '</div>' +
          '<div class="gguf-metadata-panel hidden" data-meta-panel="' + U.escapeHtml(m.id) + '"></div>';

        if (swarmBody) swarmBody.appendChild(card);

        // Restore per-model activity ticker from stored events (DOM only, don't re-log)
        if (_modelEvents[m.id] && _modelEvents[m.id].length > 0) {
          App.dashboard._renderModelTicker(m.id);
        }

        // Measure pinned endpoint rows (first + last) and set CSS custom
        // properties so the pipeline-encrypted connector line starts at the
        // first tick and ends at the last tick — not the whole right column.
        if (m.encrypted_pipeline && !isCompact) {
          requestAnimationFrame(function() {
            App.dashboard._measurePipelineConnector(card);
          });
        }
        if (!isCompact) {
          requestAnimationFrame(function() {
            App.dashboard._applyPipelinePlan(card);
          });
        }
      });

      // --- Shared helpers for cloud + subscription card rendering ---
      function getCtxLen(cm) {
        if (!cm.meta) return cm.context_length || 0;
        return cm.meta.context_length || cm.meta.context_window || cm.meta.max_model_len || cm.context_length || 0;
      }
      var _nonChatPattern = /dall-e|tts|whisper|embed|moderation|davinci-\d|babbage-\d|text-embedding|audio/i;
      function sortCloudModels(models, sortBy) {
        var sorted = models.slice();
        if (sortBy === 'ctx-desc') sorted.sort(function(a, b) { return getCtxLen(b) - getCtxLen(a); });
        else if (sortBy === 'ctx-asc') sorted.sort(function(a, b) { return getCtxLen(a) - getCtxLen(b); });
        else if (sortBy === 'avail') sorted.sort(function(a, b) {
          var sa = S.modelStatus[a.id], sb = S.modelStatus[b.id];
          var rank = { up: 0, rate_limited: 1, timeout: 3, unavailable: 4, not_found: 5, error: 4 };
          var ra = sa ? (rank[sa.status] !== undefined ? rank[sa.status] : 2) : 2;
          var rb = sb ? (rank[sb.status] !== undefined ? rank[sb.status] : 2) : 2;
          if (ra !== rb) return ra - rb;
          return (sa ? sa.latency_ms : 99999) - (sb ? sb.latency_ms : 99999);
        });
        else if (sortBy === 'popular') sorted.sort(function(a, b) {
          var aNon = _nonChatPattern.test(a.id) ? 1 : 0, bNon = _nonChatPattern.test(b.id) ? 1 : 0;
          if (aNon !== bNon) return aNon - bNon;
          var ca = (a.meta && a.meta.created) || 0, cb = (b.meta && b.meta.created) || 0;
          if (ca !== cb) return cb - ca;
          var na = (a.name || a.id).toLowerCase(), nb = (b.name || b.id).toLowerCase();
          return na < nb ? -1 : na > nb ? 1 : 0;
        });
        else sorted.sort(function(a, b) {
          var na = (a.name || a.id).toLowerCase(), nb = (b.name || b.id).toLowerCase();
          return na < nb ? -1 : na > nb ? 1 : 0;
        });
        return sorted;
      }
      function renderCloudRow(cm) {
        var ctx = getCtxLen(cm);
        var ctxStr = ctx > 0 ? (ctx >= 1000 ? Math.round(ctx / 1000) + 'K' : ctx.toString()) : '';
        var pingHtml = App.providerHealth.modelBadgeHtml(cm.id);
        return '<div class="cloud-model-row" data-select-cloud="' + U.escapeHtml(cm.id) + '" title="' + U.escapeHtml(cm.id) + '">' +
          '<span class="cloud-model-row-name">' + U.escapeHtml(cm.name || cm.id) + '</span>' +
          (ctxStr ? '<span class="cloud-model-row-ctx">' + ctxStr + '</span>' : '<span class="cloud-model-row-ctx"></span>') +
          '<span class="cloud-model-row-ping">' + pingHtml + '</span></div>';
      }
      function renderRowsInto(container, models) {
        container.innerHTML = models.length > 0
          ? models.map(renderCloudRow).join('')
          : '<div class="cloud-model-empty">' + U.escapeHtml(I18n.t('dashboard.cloud_no_match')) + '</div>';
      }
      // Skip probing non-chat endpoints (DALL-E, Whisper, embeddings, moderation)
      // — they'd always 404 and add noise.
      function probableChatModels(models) {
        return models.filter(function(cm) { return !_nonChatPattern.test(cm.id); });
      }
      function renderProviderCard(opts) {
        var p = opts.provider, pLabel = PROVIDER_NAMES[p] || p, pModels = opts.models;
        var sorted = sortCloudModels(pModels, 'popular');
        var prefix = opts.idPrefix || 'cloud';
        var filterId = prefix + '-filter-' + p, sortId = prefix + '-sort-' + p, listId = prefix + '-list-wrap-' + p;
        var card = document.createElement('div');
        // Start collapsed by default
        card.className = 'model-card cloud-model cloud-card-collapsed' + (opts.cardClass ? ' ' + opts.cardClass : '');
        card.setAttribute('data-provider', p);
        var cardIconHtml = providerIconHtml(p, 18);
        var expandToggleHtml = '<span class="cloud-expand-toggle" data-cloud-expand="' + U.escapeHtml(p) + '">&#9662;</span>';
        var modelCountHtml = '<span class="cloud-provider-count" title="' +
          U.escapeHtml(I18n.t('dashboard.cloud_model_count', { count: pModels.length })) + '">' +
          pModels.length + '</span>';
        var isSub = typeof isSubscriptionProvider === 'function' && isSubscriptionProvider(p);
        // Auth-type pill — same styling as the top provider health bar
        // (tag-sub violet for subscription, tag-api neutral grey for API key).
        var authTagHtml = isSub
          ? '<span class="ph-tag tag-sub" title="' + U.escapeHtml(I18n.t('dashboard.cloud_sub_note')) + '">' + U.escapeHtml(I18n.t('mode.subscription')) + '</span>'
          : '<span class="ph-tag tag-api" title="' + U.escapeHtml(I18n.t('dashboard.cloud_note', { provider: pLabel })) + '">' + U.escapeHtml(I18n.t('mode.api')) + '</span>';
        // Subscription cards put the auth-status badge into statusHtml (it gets
        // replaced by the CLI fetch). API-key cards have no separate status
        // badge — the tag-api pill alone conveys the auth mode.
        var statusHtml = opts.statusHtml || '';
        card.innerHTML =
          '<div class="cloud-card-header' + (opts.headerClass ? ' ' + opts.headerClass : '') + '">' +
            '<span class="cloud-provider-name">' + (cardIconHtml ? cardIconHtml + ' ' : '') + U.escapeHtml(pLabel) + modelCountHtml + '</span>' +
            '<span style="display:flex;align-items:center;gap:8px">' +
              statusHtml +
              authTagHtml +
              expandToggleHtml +
            '</span>' +
          '</div>' +
          '<div class="cloud-card-controls">' +
            '<input type="text" class="cloud-model-filter" id="' + filterId + '" placeholder="' + U.escapeHtml(I18n.t('dashboard.cloud_search')) + '" autocomplete="off">' +
            '<select class="cloud-model-sort" id="' + sortId + '">' +
              '<option value="popular">' + U.escapeHtml(I18n.t('dashboard.cloud_sort_newest')) + '</option>' +
              '<option value="az">' + U.escapeHtml(I18n.t('dashboard.sort_az')) + '</option>' +
              '<option value="ctx-desc">' + U.escapeHtml(I18n.t('dashboard.cloud_sort_ctx_desc')) + '</option>' +
              '<option value="ctx-asc">' + U.escapeHtml(I18n.t('dashboard.cloud_sort_ctx_asc')) + '</option>' +
              '<option value="avail">' + U.escapeHtml(I18n.t('dashboard.cloud_sort_ping')) + '</option>' +
            '</select>' +
          '</div>' +
          '<div class="cloud-model-list" id="' + listId + '"></div>' +
          '<div class="cloud-card-note">' + U.escapeHtml(opts.noteText || I18n.t('dashboard.cloud_note', { provider: pLabel })) + '</div>';
        opts.parentEl.appendChild(card);
        var listContainer = document.getElementById(listId);
        if (listContainer) renderRowsInto(listContainer, sorted);
        setTimeout(function() { App.providerHealth.probe(probableChatModels(sorted).slice(0, 20).map(function(cm) { return cm.id; })); }, 500);
        var filterEl = document.getElementById(filterId), sortEl = document.getElementById(sortId);
        var refreshRows = function() {
          var query = filterEl ? filterEl.value.toLowerCase().trim() : '';
          var sortBy = sortEl ? sortEl.value : 'popular';
          var filtered = query ? pModels.filter(function(cm) {
            return ((cm.name || '') + ' ' + cm.id + ' ' + (cm.meta && cm.meta.owned_by ? cm.meta.owned_by : '')).toLowerCase().indexOf(query) !== -1;
          }) : pModels;
          var s = sortCloudModels(filtered, sortBy);
          if (listContainer) renderRowsInto(listContainer, s);
          App.providerHealth.probe(probableChatModels(s).slice(0, 20).map(function(cm) { return cm.id; }));
        };
        if (filterEl) { filterEl.addEventListener('input', refreshRows); filterEl.addEventListener('paste', function() { setTimeout(refreshRows, 0); }); }
        if (sortEl) sortEl.addEventListener('change', function() {
          refreshRows();
          if (sortEl.value === 'avail') App.providerHealth.probe(probableChatModels(pModels).map(function(cm) { return cm.id; }).slice(0, 40));
        });
      }

      // --- Cloud providers (API-key + subscription unified) ---
      if (hasCloud || hasSubscription) {
        var byProvider = {};
        apiModels.forEach(function(cm) {
          var p = cm.provider || 'cloud';
          if (!byProvider[p]) byProvider[p] = [];
          byProvider[p].push(cm);
        });
        subscriptionModels.forEach(function(cm) {
          var p = cm.provider || 'subscription';
          if (!byProvider[p]) byProvider[p] = [];
          byProvider[p].push(cm);
        });

        var providerCount = Object.keys(byProvider).length;
        var totalModels = apiModels.length + subscriptionModels.length;
        var cloudSection = document.createElement('details');
        cloudSection.className = 'models-section';
        cloudSection.open = true;
        var cloudMeta = I18n.t('dashboard.providers_count', { count: providerCount, models: totalModels });
        cloudSection.innerHTML = '<summary class="models-section-header">' +
          '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" aria-hidden="true" class="models-section-logo" style="flex-shrink:0"><path d="M18 10h-1.26A8 8 0 1 0 9 20h9a5 5 0 0 0 0-10z" fill="var(--accent)"/></svg>' +
          '<span class="models-section-title">' + U.escapeHtml(I18n.t('settings.cloud_providers')) + '</span>' +
          '<span class="models-section-count">' + cloudMeta + '</span>' +
          '</summary>';
        var cloudBody = document.createElement('div');
        cloudBody.className = 'models-section-body';
        cloudSection.appendChild(cloudBody);
        list.appendChild(cloudSection);

        // Sort: subscription providers appear first (distinctive, usually fewer models)
        var providerOrder = Object.keys(byProvider).sort(function(a, b) {
          var aSub = typeof isSubscriptionProvider === 'function' && isSubscriptionProvider(a) ? 0 : 1;
          var bSub = typeof isSubscriptionProvider === 'function' && isSubscriptionProvider(b) ? 0 : 1;
          if (aSub !== bSub) return aSub - bSub;
          return a < b ? -1 : a > b ? 1 : 0;
        });

        providerOrder.forEach(function(p) {
          var isSub = typeof isSubscriptionProvider === 'function' && isSubscriptionProvider(p);
          if (isSub) {
            renderProviderCard({
              provider: p, models: byProvider[p], parentEl: cloudBody,
              cardClass: 'subscription-model-card', headerClass: 'subscription-card-header',
              statusHtml: '<span class="badge badge-claude" id="sub-status-' + p + '">' + U.escapeHtml(I18n.t('dashboard.cloud_subscription')) + '</span>',
              noteText: I18n.t('dashboard.cloud_sub_note'),
              idPrefix: 'sub',
            });
          } else {
            renderProviderCard({ provider: p, models: byProvider[p], parentEl: cloudBody });
          }
        });

        if (hasSubscription) {
          // Fetch CLI status for subscription providers (dedup-coalesced across components)
          App.data.loadClaudeSubStatus().then(function(data) {
            if (!data || data.error) return;
            var statusEl = document.getElementById('sub-status-claude_subscription');
            if (!statusEl) return;
            var pills = [];
            if (data.authenticated) {
              pills.push('<span class="ph-tag tag-ok" title="' + U.escapeHtml(I18n.t('dashboard.sub_authenticated')) + '">\u2713 ' + U.escapeHtml(I18n.t('dashboard.sub_authenticated')) + '</span>');
              if (data.subscription_type) {
                var plan = data.subscription_type.charAt(0).toUpperCase() + data.subscription_type.slice(1);
                pills.push('<span class="ph-tag tag-plan">' + U.escapeHtml(plan) + '</span>');
              }
              if (data.cli_version) {
                // claude --version returns e.g. "2.0.5 (Claude Code)" — strip the suffix
                var ver = data.cli_version.replace(/\s*\(Claude Code\)\s*$/, '').trim();
                if (ver) pills.push('<span class="ph-tag tag-ver">v' + U.escapeHtml(ver) + '</span>');
              }
            } else {
              pills.push('<span class="ph-tag tag-down" title="' + U.escapeHtml(I18n.t('dashboard.sub_not_authenticated')) + '">\u26a0 ' + U.escapeHtml(I18n.t('dashboard.sub_not_authenticated')) + '</span>');
            }
            statusEl.outerHTML = pills.join('');
          }).catch(function() {});
        }

        if (Object.keys(S.modelStatus).length > 0) App.providerHealth.updateModelBadges();
      }
    },

    // Patch a single shard row in place. Returns true if state actually changed.
    _patchShardRow: function(row, opts) {
      if (!row) return false;
      var oldState = row.getAttribute('data-state');
      var newState = opts.state;
      var glyphEl = row.querySelector('.shard-row-state-glyph');
      var statusEl = row.querySelector('.shard-row-status');
      var existing = row.querySelector('.shard-row-piecebar');
      if (oldState !== newState) {
        row.setAttribute('data-state', newState);
        row.classList.add('shard-transitioning');
        setTimeout(function() { row.classList.remove('shard-transitioning'); }, 1500);
        if (glyphEl) glyphEl.textContent = _shardGlyph(newState);
      }
      if (statusEl && opts.statusText !== undefined) statusEl.textContent = opts.statusText;

      // Piece-bar: add/update/remove to match current download state
      if (opts.peerDownloads && opts.peerDownloads.length > 0 && newState === 'downloading') {
        var newBar = _buildPieceBar(opts.peerDownloads, opts.dlPct || 0);
        if (existing) existing.outerHTML = newBar;
        else row.insertAdjacentHTML('beforeend', newBar);
      } else if (existing && newState !== 'downloading') {
        existing.remove();
      }
      return oldState !== newState;
    },

    updateShardsLive: function(acquisitions, shardRegistry, peerDownloads) {
      if (!acquisitions && !shardRegistry && !peerDownloads) return;
      var self = this;

      // Index peerDownloads by modelId/shardIndex for quick lookup during patches
      var pdIndex = {};
      if (peerDownloads && peerDownloads.length > 0) {
        peerDownloads.forEach(function(pd) {
          var k = pd.model_id + ':' + pd.shard_index;
          if (!pdIndex[k]) pdIndex[k] = [];
          pdIndex[k].push({ node_id: pd.node_id, progress_pct: pd.progress_pct || 0 });
        });
      }

      if (acquisitions) {
        acquisitions.forEach(function(acq) {
          var modelId = acq.model_id;
          if (!modelId) return;
          var safeId = U.safeId(modelId);

          var shardDetails = acq.shard_details || [];
          shardDetails.forEach(function(sd) {
            var rowId = safeId + '-' + sd.index;
            var row = document.querySelector('[data-shard-row="' + U.cssSafeAttr(rowId) + '"]');
            if (!row) return;

            var dlPct = sd.progress_pct || 0;
            var newState = 'missing';
            var statusText = I18n.t('shard.row.missing_label');
            if (sd.state === 'complete') { newState = 'disk'; statusText = I18n.t('shard.row.disk_label'); }
            else if (sd.state === 'verifying') { newState = 'downloading'; statusText = dlPct + '%\u2193'; }
            else if (sd.state === 'downloading') { newState = 'downloading'; statusText = dlPct + '%\u2193'; }
            else if (sd.state === 'pending') { newState = 'downloading'; statusText = '\u2022'; }

            self._patchShardRow(row, {
              state: newState,
              statusText: statusText,
              peerDownloads: pdIndex[modelId + ':' + sd.index],
              dlPct: dlPct,
            });
          });

          // Update progress bar
          var progressEl = document.querySelector('[data-model-progress="' + safeId + '"]');
          if (progressEl && acq.total_bytes > 0) {
            var dlBytes = Math.min(acq.downloaded_bytes || 0, acq.total_bytes);
            var pct = Math.min(100, Math.round((dlBytes / acq.total_bytes) * 100));
            var lastPct = parseInt(progressEl.getAttribute('data-last-pct') || '0', 10);
            if (pct >= lastPct) {
              progressEl.setAttribute('data-last-pct', '' + pct);
              var speed = acq.speed_bytes_per_sec || 0;
              var shardLabel = acq.downloaded_shards !== undefined ? I18n.t('dashboard.shard_progress_label', { dl: acq.downloaded_shards, total: acq.total_shards || shardDetails.length }) : I18n.t('dashboard.downloading_label');
              var etaStr = '';
              if (speed > 0 && acq.total_bytes > dlBytes) {
                etaStr = U.formatEta((acq.total_bytes - dlBytes) / speed);
              }
              var textEl = progressEl.querySelector('.dl-progress-text');
              if (textEl) {
                var txt = U.formatDlProgress(dlBytes, acq.total_bytes, pct);
                if (speed > 0) txt += ' \u00b7 ' + U.formatSpeed(speed);
                if (etaStr) txt += I18n.t('dashboard.eta', { eta: etaStr });
                textEl.textContent = txt;
              }
              var labelEl = progressEl.querySelector('.text-muted');
              if (labelEl) labelEl.textContent = shardLabel;
              var segs = progressEl.querySelectorAll('.dl-seg');
              if (segs.length > 0) {
                shardDetails.forEach(function(sd) {
                  if (segs[sd.index]) {
                    var segFill = segs[sd.index].querySelector('.dl-seg-fill');
                    var segPct = sd.state === 'complete' ? 100 : (sd.progress_pct || 0);
                    if (segFill) segFill.style.width = segPct + '%';
                  }
                });
              } else {
                var fillEl = progressEl.querySelector('.dl-fill');
                if (fillEl) fillEl.style.width = pct + '%';
              }
            }
          } else if (!progressEl && acq.total_bytes > 0 && acq.downloaded_bytes > 0) {
            var card = document.querySelector('[data-model-id="' + U.cssSafeAttr(modelId) + '"]');
            if (card && !card.querySelector('.dl-progress')) {
              var dlBytes2 = Math.min(acq.downloaded_bytes, acq.total_bytes);
              var pct2 = Math.min(100, Math.round((dlBytes2 / acq.total_bytes) * 100));
              var speed2 = acq.speed_bytes_per_sec || 0;
              var shardLabel2 = acq.downloaded_shards !== undefined ? I18n.t('dashboard.shard_progress_label', { dl: acq.downloaded_shards, total: acq.total_shards || '?' }) : I18n.t('dashboard.downloading_label');
              var rightText2 = U.formatDlProgress(dlBytes2, acq.total_bytes, pct2) + (speed2 > 0 ? ' \u2014 ' + U.formatSpeed(speed2) : '');
              var progWrapper = document.createElement('div');
              progWrapper.innerHTML = _buildProgressBar({ safeId: safeId, pct: pct2, label: shardLabel2, rightText: rightText2 });
              var progDiv = progWrapper.firstChild;
              card.appendChild(progDiv);
              if (!card.classList.contains('downloading')) {
                card.classList.remove('partial');
                card.classList.add('downloading');
              }
            }
          }

          // Update shard summary
          var summaryEl = document.querySelector('[data-model-summary="' + safeId + '"]');
          if (summaryEl && shardDetails.length > 0) {
            var summParts = [];
            if (localCount > 0) summParts.push('<span class="shard-sum-item shard-sum-local"><span class="shard-sum-dot"></span>' + I18n.t('dashboard.summary_local', { count: localCount }) + '</span>');
            if (peerCount > 0) summParts.push('<span class="shard-sum-item shard-sum-peer"><span class="shard-sum-dot"></span>' + I18n.t('dashboard.peer_count', { count: peerCount }) + '</span>');
            if (dlCount > 0) summParts.push('<span class="shard-sum-item shard-sum-dl"><span class="shard-sum-dot"></span>' + I18n.t('dashboard.summary_downloading', { count: dlCount }) + '</span>');
            if (peerDlCount > 0) summParts.push('<span class="shard-sum-item shard-sum-peer-dl"><span class="shard-sum-dot"></span>' + I18n.t('dashboard.summary_peer_dl', { count: peerDlCount }) + '</span>');
            if (queuedCount > 0) summParts.push('<span class="shard-sum-item shard-sum-queued"><span class="shard-sum-dot"></span>' + I18n.t('dashboard.summary_queued', { count: queuedCount }) + '</span>');
            if (missingCount > 0) summParts.push('<span class="shard-sum-item shard-sum-missing"><span class="shard-sum-dot"></span>' + I18n.t('dashboard.summary_missing', { count: missingCount }) + '</span>');
            summaryEl.innerHTML = summParts.join('');
          }
        });
      }

      // Patch shard rows from shardRegistry (peer availability snapshot)
      if (shardRegistry) {
        Object.keys(shardRegistry).forEach(function(modelId) {
          var safeId = U.safeId(modelId);
          var shards = shardRegistry[modelId] || [];
          shards.forEach(function(s) {
            var rowId = safeId + '-' + s.index;
            var row = document.querySelector('[data-shard-row="' + U.cssSafeAttr(rowId) + '"]');
            if (!row) return;
            var current = row.getAttribute('data-state') || 'missing';
            if (current === 'downloading') return;

            var pdKey = modelId + ':' + s.index;
            if (pdIndex[pdKey]) {
              // Active peer download — force downloading state
              var pct0 = pdIndex[pdKey][0] ? pdIndex[pdKey][0].progress_pct : 0;
              self._patchShardRow(row, {
                state: 'downloading',
                statusText: (pct0 || 0) + '%\u2193',
                peerDownloads: pdIndex[pdKey],
                dlPct: pct0,
              });
              return;
            }

            var newState;
            var statusText;
            if (s.local && s.in_vram) { newState = 'vram'; statusText = I18n.t('shard.row.vram_label'); }
            else if (s.local) { newState = 'disk'; statusText = I18n.t('shard.row.disk_label'); }
            else if (s.holders > 0) { newState = 'peer'; statusText = I18n.t('shard.row.peer_label'); }
            else { newState = 'missing'; statusText = I18n.t('shard.row.missing_label'); }

            self._patchShardRow(row, { state: newState, statusText: statusText });
          });
        });
      }

      // Peer downloads without accompanying registry entry — patch row as downloading
      if (peerDownloads && peerDownloads.length > 0) {
        peerDownloads.forEach(function(pd) {
          var safeId = U.safeId(pd.model_id);
          var rowId = safeId + '-' + pd.shard_index;
          var row = document.querySelector('[data-shard-row="' + U.cssSafeAttr(rowId) + '"]');
          if (!row) return;
          var cur = row.getAttribute('data-state');
          if (cur === 'vram' || cur === 'disk') return;
          var pct = pd.progress_pct || 0;
          self._patchShardRow(row, {
            state: 'downloading',
            statusText: pct + '%\u2193',
            peerDownloads: pdIndex[pd.model_id + ':' + pd.shard_index] || [pd],
            dlPct: pct,
          });
        });
      }
    },

    renderPeerItem: function(p) {
      var tmpl = document.getElementById('tmpl-peer-row');
      if (!tmpl) return document.createElement('div');
      var node = tmpl.content.cloneNode(true);
      var div = node.querySelector('.peer-row-item');

      var dot = node.querySelector('.status-dot');
      dot.classList.add(p.healthy ? 'online' : 'degraded');

      var lanBadge = node.querySelector('.peer-lan-badge');
      if (p.is_lan_peer) {
        lanBadge.removeAttribute('hidden');
        lanBadge.textContent = I18n.t('dashboard.lan_badge');
      }

      var label = node.querySelector('.peer-label');
      if (p.nickname) {
        label.textContent = p.nickname;
        var sub = document.createElement('span');
        sub.className = 'text-muted mono';
        sub.style.fontSize = '0.65rem';
        sub.textContent = ' (' + (p.node_id || '').substring(0, 8) + ')';
        label.appendChild(sub);
      } else {
        label.className = 'peer-label mono';
        label.textContent = (p.node_id || 'unknown').substring(0, 16);
      }

      var gpu = node.querySelector('.peer-gpu');
      if (p.gpu) {
        gpu.textContent = p.gpu;
      } else {
        gpu.remove();
      }

      return div;
    },

    _peerSort: 'shards',
    _peerSortDir: 'desc',

    renderPeers: function(peers) {
      var list = document.getElementById('peers-list');
      var summary = document.getElementById('peers-summary');
      var overflow = document.getElementById('peers-overflow');
      var pLoading = document.getElementById('peers-loading');
      if (pLoading) pLoading.remove();
      if (!list) return;
      if (overflow) overflow.style.display = 'none';

      App.dashboard._lastPeers = peers || [];

      if (!peers || peers.length === 0) {
        if (summary) summary.textContent = '';
        list.innerHTML = '<div class="empty-state" style="padding:16px 0"><div class="empty-icon">\u{1F310}</div><p>' + I18n.t('network.no_peers_yet') + '</p></div>';
        return;
      }

      var lanCount = peers.filter(function(p) { return p.is_lan_peer; }).length;
      var healthyCount = peers.filter(function(p) { return p.healthy; }).length;
      if (summary) {
        summary.textContent = I18n.t('dashboard.peers_summary', { count: peers.length, lan: lanCount, healthy: healthyCount });
      }

      // Sort peers
      var sortKey = App.dashboard._peerSort;
      var sortDir = App.dashboard._peerSortDir;
      var sorted = peers.slice().sort(function(a, b) {
        var va, vb;
        if (sortKey === 'name') {
          va = (a.nickname || a.node_id || '').toLowerCase();
          vb = (b.nickname || b.node_id || '').toLowerCase();
          return sortDir === 'asc' ? (va < vb ? -1 : va > vb ? 1 : 0) : (va > vb ? -1 : va < vb ? 1 : 0);
        }
        if (sortKey === 'latency') { va = a.latency_ms || 99999; vb = b.latency_ms || 99999; }
        else if (sortKey === 'shards') { va = a.hosted_shards || 0; vb = b.hosted_shards || 0; }
        else if (sortKey === 'trust') { va = a.trust_score || 0; vb = b.trust_score || 0; }
        else if (sortKey === 'credits') { va = a.credits || 0; vb = b.credits || 0; }
        else { va = a.healthy ? 1 : 0; vb = b.healthy ? 1 : 0; }
        return sortDir === 'asc' ? va - vb : vb - va;
      });

      // Render as sortable table
      function _sortArrow(key) {
        var isSorted = sortKey === key;
        var arrow = sortDir === 'asc' ? '\u25B2' : '\u25BC';
        return '<span class="sort-arrow">' + (isSorted ? arrow : '\u25B4') + '</span>';
      }
      function _thClass(key) { return sortKey === key ? ' class="sorted"' : ''; }

      var html = '<table class="peer-table"><thead><tr>' +
        '<th data-peer-sort="name"' + _thClass('name') + '>' + U.escapeHtml(I18n.t('dashboard.peer_col_name')) + _sortArrow('name') + '</th>' +
        '<th data-peer-sort="latency"' + _thClass('latency') + '>' + U.escapeHtml(I18n.t('dashboard.peer_col_latency')) + _sortArrow('latency') + '</th>' +
        '<th data-peer-sort="shards"' + _thClass('shards') + '>' + U.escapeHtml(I18n.t('dashboard.peer_col_shards')) + _sortArrow('shards') + '</th>' +
        '<th data-peer-sort="trust"' + _thClass('trust') + '>' + U.escapeHtml(I18n.t('dashboard.peer_col_trust')) + _sortArrow('trust') + '</th>' +
        '<th data-peer-sort="status"' + _thClass('status') + '>' + U.escapeHtml(I18n.t('dashboard.peer_col_status')) + _sortArrow('status') + '</th>' +
        '</tr></thead><tbody>';

      sorted.forEach(function(p) {
        var name = p.nickname || (p.node_id || 'unknown').substring(0, 12);
        var idSub = p.nickname ? '<span class="peer-id-sub">' + (p.node_id || '').substring(0, 8) + '</span>' : '';
        var lanBadge = p.is_lan_peer ? ' <span class="badge badge-purple lan-badge">' + U.escapeHtml(I18n.t('dashboard.lan_badge')) + '</span>' : '';
        var dotClass = p.healthy ? 'online' : 'degraded';
        var latency = p.latency_ms ? p.latency_ms + 'ms' : '\u2014';
        var shards = p.hosted_shards || 0;
        var trust = p.trust_score !== undefined ? (p.trust_score * 100).toFixed(0) + '%' : '\u2014';
        var status = p.healthy ? I18n.t('dashboard.peer_healthy') : I18n.t('dashboard.peer_degraded');
        var gpu = p.gpu ? '<div class="text-muted" style="font-size:0.62rem">' + U.escapeHtml(p.gpu) + '</div>' : '';

        html += '<tr>' +
          '<td><div class="peer-name-cell"><span class="status-dot ' + dotClass + '"></span><span class="peer-nick">' + U.escapeHtml(name) + '</span>' + idSub + lanBadge + '</div>' + gpu + '</td>' +
          '<td class="mono">' + latency + '</td>' +
          '<td class="mono">' + shards + '</td>' +
          '<td class="mono">' + trust + '</td>' +
          '<td><span class="status-dot ' + dotClass + '" style="display:inline-block;vertical-align:middle;margin-right:4px"></span>' + U.escapeHtml(status) + '</td>' +
          '</tr>';
      });

      html += '</tbody></table>';
      list.innerHTML = html;
    },

    loadNetworkData: async function() {
      try {
        var peers = await App.data.loadPeers();
        App.dashboard.renderPeers(peers);
      } catch (e) {
        var list = document.getElementById('peers-list');
        var pLoading2 = document.getElementById('peers-loading');
        if (pLoading2) pLoading2.remove();
        if (list) list.innerHTML = '<div class="empty-state" style="padding:16px 0"><div class="empty-icon">\u{1F310}</div><p>' + I18n.t('network.no_peers_yet') + '</p></div>';
      }
    },

    updateAcquisitionProgress: function(acquisitions) {
      if (!acquisitions || acquisitions.length === 0) return;
      acquisitions.forEach(function(status) {
        var modelId = status.model_id;
        if (!modelId) return;
        if (!S.activeAcquisitions[modelId]) {
          // Skip stale complete/failed entries we aren't tracking
          var isFailed = status.state === 'failed' || (typeof status.state === 'object' && status.state && status.state.failed);
          if (status.state === 'complete' || isFailed || status.overall_pct >= 100) {
            return;
          }
          S.activeAcquisitions[modelId] = { started: Date.now() };
        }
        // Skip if already completed/failed — don't re-render the progress bar
        if (S.activeAcquisitions[modelId]._completeFired || S.activeAcquisitions[modelId]._failFired) {
          return;
        }
        App.dashboard.renderAcquisitionPanel(modelId, status);

        // Detect completion: explicit state OR all tracked shards at 100%
        var isComplete = status.state === 'complete';
        if (!isComplete && status.shard_details && status.shard_details.length > 0) {
          isComplete = status.shard_details.every(function(sd) { return sd.state === 'complete'; });
        }
        if (!isComplete && status.overall_pct >= 100) {
          isComplete = true;
        }

        // Remove download bar immediately on complete or fail
        function _removeDownloadBar(mid) {
          var safeId2 = U.safeId(mid);
          var progBar = document.querySelector('[data-model-progress="' + safeId2 + '"]');
          if (progBar) progBar.remove();
          var card2 = document.querySelector('[data-model-id="' + U.cssSafeAttr(mid) + '"]');
          if (card2) card2.classList.remove('downloading');
        }

        if (isComplete && !S.activeAcquisitions[modelId]._completeFired) {
          S.activeAcquisitions[modelId]._completeFired = true;
          // Flash "Download complete" then fade out
          var safeIdC = U.safeId(modelId);
          var progBarC = document.querySelector('[data-model-progress="' + safeIdC + '"]');
          if (progBarC) {
            progBarC.innerHTML = '<div class="dl-complete-flash">' + U.escapeHtml(I18n.t('dashboard.download_complete')) + '</div>';
            progBarC.classList.add('dl-complete');
            setTimeout(function() { _removeDownloadBar(modelId); }, 3000);
          }
          // Toast handled by backend activity_event (model_download_complete / hf_download_complete)
          // Keep activeAcquisitions entry with _completeFired flag for 30s so incoming
          // stats_update messages don't re-create the download bar (backend removes
          // acquisition_progress after 5s, but WS messages can arrive in between)
          setTimeout(function() { App.dashboard.loadInitial(); }, 3500);
          setTimeout(function() { delete S.activeAcquisitions[modelId]; }, 30000);
        } else if (!isComplete && (status.state === 'failed' || (typeof status.state === 'object' && status.state && status.state.failed)) && !S.activeAcquisitions[modelId]._failFired) {
          S.activeAcquisitions[modelId]._failFired = true;
          _removeDownloadBar(modelId);
          var reason = (typeof status.state === 'object' && status.state.failed) ? (status.state.failed.reason || '') : '';
          // Toast handled by backend activity_event (shard_download_failed / hf_download_failed)
          setTimeout(function() { delete S.activeAcquisitions[modelId]; }, 5000);
        }
      });
    },

    renderAcquisitionPanel: function(modelId, status) {
      if (!status) return;
      if (!S.activeAcquisitions[modelId]) return;
      var safeId = U.safeId(modelId);
      var card = document.querySelector('[data-model-id="' + U.cssSafeAttr(modelId) + '"]');
      if (!card) {
        App.models.load();
        App.dashboard.loadInitial();
        return;
      }

      var stateName = typeof status.state === 'string' ? status.state : 'unknown';

      if (stateName === 'complete') {
        if (!card.classList.contains('ready')) {
          setTimeout(function() { App.dashboard.loadInitial(); }, 1500);
        }
        return;
      }

      if (!card.classList.contains('downloading')) {
        card.classList.remove('partial');
        card.classList.add('downloading');
      }

      var totalBytes = status.total_bytes || 0;
      var dlBytes = status.downloaded_bytes || 0;
      var pct = totalBytes > 0 ? Math.round((dlBytes / totalBytes) * 100) : 0;
      var speed = status.speed_bytes_per_sec || 0;

      var progressEl = card.querySelector('.dl-progress');
      if (!progressEl) {
        progressEl = document.createElement('div');
        progressEl.className = 'dl-progress';
        progressEl.setAttribute('data-model-progress', safeId);
        card.appendChild(progressEl);
      }

      var speedStr = speed > 0 ? ' - ' + U.formatSpeed(speed) : '';
      var cancelBtn = '<button class="btn btn-sm" style="padding:1px 6px;font-size:0.7rem;line-height:1.2" data-cancel-download="' + U.escapeHtml(modelId) + '" title="' + U.escapeHtml(I18n.t('shard.cancel_download')) + '">&times; ' + U.escapeHtml(I18n.t('actions.cancel')) + '</button>';
      var rightText3 = U.formatDlProgress(dlBytes, totalBytes, pct) + speedStr;
      var wrapper = document.createElement('div');
      wrapper.innerHTML = _buildProgressBar({ safeId: safeId, pct: pct, label: U.escapeHtml(I18n.t('dashboard.downloading_data')), rightText: rightText3, cancelBtn: cancelBtn });
      progressEl.innerHTML = wrapper.firstChild.innerHTML;

      var oldPanel = document.getElementById('acq-panel-' + safeId);
      if (oldPanel) oldPanel.remove();
    }
  };
})();
