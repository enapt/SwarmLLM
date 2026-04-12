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

  function _shardReplicaPips(s) {
    var holders = s.holders || 0;
    var tier = holders >= 3 ? 'good' : holders >= 1 ? 'low' : 'none';
    var titleKey = holders === 0 ? 'shard.row.replicas_none'
                 : (holders === 1 ? 'shard.row.replicas_count_one' : 'shard.row.replicas_count_other');
    var title = I18n.t(titleKey, { n: holders });
    if (holders === 0) {
      return '<span class="shard-row-replicas" data-tier="none" title="' + U.escapeHtml(title) + '">' +
        '<span class="shard-row-pip"></span></span>';
    }
    var visible = Math.min(holders, 4);
    var html = '';
    for (var i = 0; i < visible; i++) html += '<span class="shard-row-pip"></span>';
    if (holders > 4) html += '<span class="shard-row-pip-more">+' + (holders - 4) + '</span>';
    return '<span class="shard-row-replicas" data-tier="' + tier + '" title="' + U.escapeHtml(title) + '">' + html + '</span>';
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
    var layerRange = '';
    var statusLabel = _shardStatusLabel(s, state);
    var sizeText = s.size_bytes ? U.formatBytes(s.size_bytes) : '\u2014';
    var lockCls = s.locked ? ' locked' : '';
    var lockGlyph = s.locked ? '\uD83D\uDD12' : '\uD83D\uDD13';
    var lockTitle = s.locked ? I18n.t('shard.unlock') : I18n.t('shard.lock');
    var pieceBar = (state === 'downloading' && s.peer_downloads && s.peer_downloads.length > 0)
      ? _buildPieceBar(s.peer_downloads, (s.download && s.download.progress_pct) || 0)
      : '';
    var actions = _buildRowActions(state, !!s.local, !!s.in_vram);
    return '<div class="shard-row" data-state="' + state + '"' +
      ' data-shard-row="' + safeId + '-' + s.index + '"' +
      ' data-shard-model="' + U.escapeHtml(m.id) + '"' +
      ' data-shard-index="' + s.index + '"' +
      ' data-shard-locked="' + (s.locked ? '1' : '0') + '">' +
      '<span class="shard-row-state-glyph">' + _shardGlyph(state) + '</span>' +
      '<span class="shard-row-index">' + idxLabel + '</span>' +
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

    // Column headers: show every shard index; cap density if many
    var headHtml = '<tr><th></th>';
    var colEvery = shards.length > 40 ? 5 : 1;
    shards.forEach(function(s, i) {
      var isMmproj = s.index === MMPROJ_SHARD_INDEX;
      var label = isMmproj ? '\u2605' : ((i % colEvery === 0) ? String(s.index + 1) : '');
      headHtml += '<th title="' + U.escapeHtml(I18n.t('shard.matrix.col_header_tip', { n: s.index + 1 }) || '') + '">' + label + '</th>';
    });
    headHtml += '</tr>';

    // Self row
    var selfRow = '<tr><th class="you" title="' + U.escapeHtml(m.id) + '">' +
      U.escapeHtml(I18n.t('shard.matrix.peer_you')) + '</th>';
    shards.forEach(function(s) {
      var state = _shardState(s);
      // Self-row "peer" state is effectively absent locally — but holders count > 0 could mean
      // other peers have it. For self-row we want local presence only.
      if (state === 'peer') state = 'absent';
      var glyph = state === 'vram' ? '\u25A0' : state === 'disk' ? '\u25A1'
                : state === 'downloading' ? '\u25D0' : '';
      selfRow += '<td data-state="' + state + '">' + glyph + '</td>';
    });
    selfRow += '</tr>';

    // Peer rows
    var peerRows = capped.map(function(entry) {
      var pid = entry.pid;
      var shortId = pid.length > 8 ? pid.substring(0, 8) : pid;
      var swatch = '<span class="srm-peer-swatch" style="background:' + U.peerColor(pid) + '"></span>';
      var row = '<tr><th title="' + U.escapeHtml(pid) + '">' + swatch + U.escapeHtml(shortId) + '</th>';
      shards.forEach(function(s) {
        var has = (s.holder_ids || []).indexOf(pid) !== -1;
        var state = has ? 'disk' : 'absent';
        var glyph = has ? '\u25A1' : '';
        row += '<td data-state="' + state + '">' + glyph + '</td>';
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

    return '<div class="shard-matrix" data-shard-matrix="' + safeId + '"' + (showAll ? ' data-expanded="1"' : '') + '>' +
      '<table>' +
      '<thead>' + headHtml + '</thead>' +
      '<tbody>' + selfRow + peerRows + '</tbody>' +
      '</table>' +
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
      });
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
            } else {
              if (vramLabel) vramLabel.textContent = I18n.t('hw.vram_idle');
              // CPU mode: show actual GPU VRAM (driver baseline only, models use RAM)
              vramEl.textContent = U.formatMB(vramUsed) + ' / ' + U.formatMB(vramTotal);
              vramEl.title = I18n.t('hw.vram_idle_tip');
              var vramPct = vramTotal > 0 ? (vramUsed / vramTotal * 100) : 0;
              document.getElementById('vram-bar').style.width = vramPct.toFixed(1) + '%';
              document.getElementById('vram-bar').className = 'fill cyan';
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
        card.className = 'model-card' + (isReady ? ' ready' : (isDownloading ? ' downloading' : (isPartial ? ' partial' : ''))) + (isCompact ? ' compact' : '');
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
        // Encrypted pipeline badge
        if (m.shard_count > 1 && m.local) {
          var encReady = m.has_first_shard && m.has_last_shard;
          var encActive = m.encrypted_pipeline;
          var encClass = encActive ? 'badge-encrypted active' : (encReady ? 'badge-encrypted ready' : 'badge-encrypted faded');
          var encTitle = encActive ? I18n.t('dashboard.enc_active') :
            (encReady ? I18n.t('dashboard.enc_available') : I18n.t('dashboard.enc_unavailable'));
          var missingParts = [];
          if (!m.has_first_shard) missingParts.push(I18n.t('dashboard.enc_missing_first'));
          if (!m.has_last_shard) missingParts.push(I18n.t('dashboard.enc_missing_last', { n: m.shard_count - 1 }));
          if (missingParts.length > 0) encTitle += '. ' + I18n.t('dashboard.enc_missing', { parts: missingParts.join(', ') });
          detailParts.push('<span class="' + encClass + '" data-enc-toggle="' + U.escapeHtml(m.id) + '" data-enc-ready="' + (encReady ? '1' : '0') + '" title="' + U.escapeHtml(encTitle) + '">&#128274;</span>');
        }
        // Source label
        if (m.source === 'network' && hostedShards === 0) {
          detailParts.push('<span class="badge badge-remote" title="' + U.escapeHtml(I18n.t('dashboard.badge_remote')) + '">' + U.escapeHtml(I18n.t('dashboard.badge_remote_label')) + '</span>');
        }
        if (detailParts.length > 0) {
          detailBadgesHtml = '<div class="model-card-detail-badges">' + detailParts.join('') + '</div>';
        }

        // Gear + info buttons
        var gearHtml = '<button class="model-gear-btn" data-am-gear="' + U.escapeHtml(m.id) + '" title="' + U.escapeHtml(I18n.t('dashboard.gear_title')) + '">&#9881;</button>';
        var metaBtnHtml = m.has_header ? '<button class="model-meta-btn" data-meta-toggle="' + U.escapeHtml(m.id) + '" title="' + U.escapeHtml(I18n.t('models.metadata_header')) + '">&#9432;</button>' : '';

        // Shard grid (legacy) — retained only to build the health summary badge that
        // appears elsewhere in the card. The actual expanded rendering uses the new
        // two-column shard-list builder below.
        var shardHtml = '';
        var healthBadgeHtml = '';
        var healthBarHtml = '';
        if (shards.length > 0) {
          var lastIdx = shardCount - 1;
          var sizeClass = shardCount > 50 ? ' shard-grid-sm' : (shardCount > 20 ? ' shard-grid-md' : '');
          shardHtml = '<div class="shard-grid' + sizeClass + '" role="grid" aria-label="' + U.escapeHtml(I18n.t('dashboard.shard_grid_aria', { model: U.formatModelDisplayName(m.name || m.id) })) + '" data-model-grid="' + safeId + '">';
          var localCount = 0, peerCount = 0, dlCount = 0, peerDlCount = 0, queuedCount = 0, missingCount = 0;

          shards.forEach(function(s) {
            var cls = 'missing';
            var label = '' + (s.index + 1);
            var dlPct = 0;

            var holderBadge = '';
            if (s.local && s.in_vram) { cls = 'local vram'; localCount++; }
            else if (s.local) { cls = 'local'; localCount++; }
            else if (s.holders > 0) {
              cls = 'peer'; peerCount++;
              holderBadge = '<span class="shard-holders">' + s.holders + '</span>';
            }
            else { missingCount++; }

            if (s.download && s.download.state === 'Downloading') {
              dlPct = s.download.progress_pct || 0;
              cls = 'downloading'; dlCount++;
              label = dlPct + '%';
              if (missingCount > 0) missingCount--;
              if (peerCount > 0 && !s.local) peerCount--;
            } else if (s.download && s.download.state === 'Verifying') {
              cls = 'verifying'; dlCount++;
              label = '\u2713';
              if (missingCount > 0) missingCount--;
              if (peerCount > 0 && !s.local) peerCount--;
            } else if (s.download && (s.download.state === 'Queued' || s.download.state === 'pending')) {
              cls = 'queued'; queuedCount++;
              label = '\u2022';
              if (missingCount > 0) missingCount--;
              if (peerCount > 0 && !s.local) peerCount--;
            }

            if (s.peer_downloads && s.peer_downloads.length > 0) {
              if (cls !== 'local' && cls !== 'downloading' && cls !== 'verifying') {
                dlPct = s.peer_downloads[0].progress_pct || 0;
                cls = 'peer-downloading'; peerDlCount++;
                label = dlPct + '%';
                if (missingCount > 0) missingCount--;
                if (peerCount > 0) peerCount--;
              }
            }

            var title = I18n.t('shard.part_n', { n: s.index + 1 }) + (s.size_bytes ? ' (' + U.formatBytes(s.size_bytes) + ')' : '');
            if (cls === 'local vram') title += ' \u2014 ' + I18n.t(S._gpuInference ? 'shard.tooltip_active_vram' : 'shard.tooltip_active_ram');
            else if (cls === 'local') title += ' \u2014 ' + I18n.t('shard.tooltip_on_disk');
            else if (cls === 'peer') title += ' \u2014 ' + I18n.t('shard.tooltip_peer_available', { count: s.holders });
            else if (cls === 'downloading') title += ' \u2014 ' + I18n.t('shard.tooltip_downloading', { pct: dlPct });
            else if (cls === 'verifying') title += ' \u2014 ' + I18n.t('shard.tooltip_verifying');
            else if (cls === 'peer-downloading') title += ' \u2014 ' + I18n.t('shard.tooltip_peer_downloading', { pct: dlPct });
            else title += ' \u2014 ' + I18n.t('shard.tooltip_unavailable');
            title += '\n' + I18n.t('shard.tooltip_click');

            var style = '';
            if (cls === 'downloading' || cls === 'peer-downloading') {
              style = ' style="--dl-pct:' + dlPct + '%"';
            }

            var lockIcon = s.locked ? '<span class="shard-lock-icon" title="' + U.escapeHtml(I18n.t('shard.locked_tooltip')) + '">\uD83D\uDD12</span>' : '';

            var endpointClass = '';
            var endpointLabel = '';
            if (shardCount > 1 && s.index === 0) {
              endpointClass = (m.encrypted_pipeline && s.local) ? ' shard-pinned' : ' shard-endpoint';
              endpointLabel = '<span class="shard-endpoint-tag">' + U.escapeHtml(I18n.t('shard.endpoint_first')) + '</span>';
            } else if (shardCount > 1 && s.index === lastIdx) {
              endpointClass = (m.encrypted_pipeline && s.local) ? ' shard-pinned' : ' shard-endpoint';
              endpointLabel = '<span class="shard-endpoint-tag">' + U.escapeHtml(I18n.t('shard.endpoint_last')) + '</span>';
            }

            shardHtml += '<div class="shard-cell ' + cls + (s.locked ? ' locked' : '') + endpointClass + '"' +
              (style ? style : '') +
              ' data-shard="' + safeId + '-' + s.index + '"' +
              ' data-shard-model="' + U.escapeHtml(m.id) + '"' +
              ' data-shard-index="' + s.index + '"' +
              ' data-shard-locked="' + (s.locked ? '1' : '0') + '"' +
              ' role="gridcell"' +
              ' aria-label="' + U.escapeHtml(title) + '"' +
              ' title="' + U.escapeHtml(title) + '">' + label + holderBadge + endpointLabel + lockIcon + '</div>';
          });
          shardHtml += '</div>';

          // Compact legend — only show entries relevant to this model's current state
          var hasVram = shards.some(function(s) { return s.local && s.in_vram; });
          var hasLocalNotVram = shards.some(function(s) { return s.local && !s.in_vram; });
          var hasPeer = peerCount > 0;
          var hasDl = dlCount > 0 || peerDlCount > 0 || queuedCount > 0;
          var hasMissing = missingCount > 0;

          var legendParts = [];
          if (hasLocalNotVram) legendParts.push('<span class="sleg"><span class="sleg-swatch sleg-local"></span>' + U.escapeHtml(I18n.t('dashboard.shard_on_pc')) + '</span>');
          if (hasVram) legendParts.push('<span class="sleg"><span class="sleg-swatch sleg-vram"></span>' + U.escapeHtml(I18n.t('dashboard.active_in', { mem: I18n.t(S._gpuInference ? 'hw.vram' : 'hw.ram') })) + '</span>');
          if (hasPeer) legendParts.push('<span class="sleg"><span class="sleg-swatch sleg-peer"></span>' + U.escapeHtml(I18n.t('dashboard.shard_on_peers')) + '</span>');
          if (hasDl) legendParts.push('<span class="sleg"><span class="sleg-swatch sleg-dl"></span>' + U.escapeHtml(I18n.t('dashboard.shard_downloading')) + '</span>');
          if (hasMissing) legendParts.push('<span class="sleg"><span class="sleg-swatch sleg-missing"></span>' + U.escapeHtml(I18n.t('dashboard.shard_missing')) + '</span>');
          if (legendParts.length > 0) {
            shardHtml += '<div class="shard-legend-bar">' + legendParts.join('') + '</div>';
          }

          // Swarm health bar — shows NETWORK replication, not local status
          // Color = holder count per shard across all nodes (including this one)
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

          // Health label based on network replication quality
          var healthLabel, healthClass;
          if (networkMissing === totalShards) { healthLabel = I18n.t('dashboard.health_unavailable_label'); healthClass = 'health-low'; }
          else if (networkMissing > 0) { healthLabel = I18n.t('dashboard.health_incomplete'); healthClass = 'health-low'; }
          else if (fragile > 0) { healthLabel = I18n.t('dashboard.health_fragile'); healthClass = 'health-partial'; }
          else if (avgHolders >= 2) { healthLabel = I18n.t('dashboard.health_healthy'); healthClass = 'health-full'; }
          else { healthLabel = I18n.t('dashboard.health_good'); healthClass = 'health-good'; }

          // Health tooltip — scale-aware language
          var barTooltipLines = [];
          var maxH = 0;
          shards.forEach(function(s) { if ((s.holders || 0) > maxH) maxH = s.holders || 0; });

          if (healthClass === 'health-full') {
            barTooltipLines.push(I18n.t('dashboard.health_well_replicated'));
          } else if (healthClass === 'health-good') {
            barTooltipLines.push(I18n.t('dashboard.health_lightly_replicated'));
          } else if (healthClass === 'health-partial') {
            barTooltipLines.push(I18n.t('dashboard.health_single_copy'));
          } else {
            barTooltipLines.push(I18n.t('dashboard.health_parts_missing'));
          }

          barTooltipLines.push('');
          if (wellReplicated > 0) barTooltipLines.push(I18n.t('dashboard.health_nodes_3plus', { count: wellReplicated, total: totalShards, nodes: maxH >= 3 ? maxH + '+' : '3+' }));
          if (adequate > 0) barTooltipLines.push(I18n.t('dashboard.health_nodes_2', { count: adequate, total: totalShards }));
          if (fragile > 0) barTooltipLines.push(I18n.t('dashboard.health_nodes_1', { count: fragile, total: totalShards }));
          if (networkMissing > 0) barTooltipLines.push(I18n.t('dashboard.health_unavailable', { count: networkMissing, total: totalShards }));
          barTooltipLines.push('');
          barTooltipLines.push(I18n.t('dashboard.health_summary', { parts: totalShards, avg: avgHolders.toFixed(1) }));

          var barTooltip = barTooltipLines.join('\n');

          // Build segmented health bar using flex — same layout as shard grid so they align
          var healthPct = totalShards > 0 ? Math.round(((totalShards - networkMissing) / totalShards) * 100) : 0;
          var barSegments = '';
          shards.forEach(function(s, i) {
            var holders = s.holders || 0;
            var color;
            if (holders >= 3) color = 'var(--green)';
            else if (holders === 2) color = 'rgba(134,239,172,0.7)';
            else if (holders === 1) color = 'var(--orange)';
            else color = 'var(--red, #ef4444)';
            barSegments += '<div class="health-seg" style="flex:1;background:' + color + '" title="' + U.escapeHtml(I18n.t('dashboard.health_seg_title', { n: i + 1, holders: holders })) + '"></div>';
          });
          var barHtml = '<div class="shard-health-bar" role="progressbar" aria-valuenow="' + healthPct + '" aria-valuemin="0" aria-valuemax="100" aria-label="' + U.escapeHtml(I18n.t('dashboard.health_bar_label', { label: healthLabel })) + '" title="' + U.escapeHtml(barTooltip) + '">' + barSegments + '</div>';

          // Summary: scale-aware detail
          var healthDetail = '';
          if (healthClass === 'health-full') {
            healthDetail = I18n.t('dashboard.health_replicated', { avg: avgHolders.toFixed(1) });
          } else if (healthClass === 'health-good') {
            healthDetail = I18n.t('dashboard.health_distributed', { count: totalShards });
          } else if (fragile > 0) {
            healthDetail = I18n.t('dashboard.health_under_replicated', { count: fragile });
          } else if (networkMissing === totalShards) {
            healthDetail = I18n.t('dashboard.health_no_shards_available');
          } else if (networkMissing > 0) {
            healthDetail = I18n.t('dashboard.health_missing', { count: networkMissing });
          }
          var healthSummary = '<div class="shard-health-summary ' + healthClass + '">' +
            '<span class="shard-health-label">' + healthLabel + '</span>' +
            '<span class="shard-health-detail">' + healthDetail + '</span>' +
            '</div>';

          // Split: badge goes in title, bar goes above shard grid
          var healthBadgeHtml = healthSummary;
          var healthBarHtml = barHtml;
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

        // --- Structured footer metadata (icon+label pairs with dot separators) ---
        var metaParts = [];
        // Size
        metaParts.push('<span class="meta-item"><span class="meta-icon">\u25A3</span>' + U.formatBytes(m.total_size_bytes || 0) + '</span>');
        // Shard count
        if (shardCount > 0) {
          metaParts.push('<span class="meta-item"><span class="meta-icon">\u2B22</span>' + I18n.t(shardCount === 1 ? 'dashboard.shard_count_one' : 'dashboard.shard_count_other', { count: shardCount }) + '</span>');
        }
        // VRAM fit indicator
        if (m.estimated_vram_mb && S._gpuInference) {
          var totalVram = (App.data.cache.stats && App.data.cache.stats.hardware && App.data.cache.stats.hardware.gpu_vram_mb) || 0;
          var fitClass = 'fit-no', fitLabel = U.formatMB(m.estimated_vram_mb);
          if (totalVram > 0) {
            var ratio = m.estimated_vram_mb / totalVram;
            if (ratio <= 0.85) { fitClass = 'fit-yes'; fitLabel = '\u2713 ' + fitLabel; }
            else if (ratio <= 1.05) { fitClass = 'fit-tight'; fitLabel = '\u2248 ' + fitLabel; }
            else { fitClass = 'fit-no'; fitLabel = '\u2717 ' + fitLabel; }
          }
          metaParts.push('<span class="meta-item"><span class="vram-fit ' + fitClass + '" title="' + U.escapeHtml(I18n.t('dashboard.vram_fit_tip', { est: U.formatMB(m.estimated_vram_mb), total: totalVram > 0 ? U.formatMB(totalVram) : '?' })) + '">' + fitLabel + '</span></span>');
        } else if (m.estimated_vram_mb && !S._gpuInference) {
          metaParts.push('<span class="meta-item" title="' + U.escapeHtml(I18n.t('hw.low_ram_tip')) + '" style="cursor:help"><span class="meta-icon">\u26A0</span><span class="meta-ok">' + U.escapeHtml(I18n.t('hw.low_ram')) + '</span></span>');
        }
        // Peer count or "Local only" warning
        if (m.peers_hosting > 0) {
          metaParts.push('<span class="meta-item"><span class="meta-icon">\u2B65</span>' + I18n.t('dashboard.peer_count', { count: m.peers_hosting }) + '</span>');
        } else if (hostedShards > 0) {
          metaParts.push('<span class="meta-item meta-warn" title="' + U.escapeHtml(I18n.t('dashboard.local_only_tip')) + '"><span class="meta-icon">\u26A0</span>' + U.escapeHtml(I18n.t('dashboard.local_only')) + '</span>');
        }
        var footerMetaHtml = metaParts.join('<span class="meta-sep">\u00B7</span>');

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
          removeHtml = '<button class="btn-action btn-danger" data-remove-model="' + U.escapeHtml(m.id) + '">' + U.escapeHtml(I18n.t('actions.remove')) + '</button>';
        }

        var name = U.formatModelDisplayName(m.name || m.id);
        var creatorIconHtml = providerIconHtml(modelIconKey(m.id), 14);
        var chevronHtml = '<span class="model-expand-chevron" title="' + U.escapeHtml(I18n.t('dashboard.expand_collapse')) + '">&#9662;</span>';

        // Active loaded class for pulsing border
        if (m.status === 'loaded') card.classList.add('active-loaded');

        // Card HTML — compact by default with availability bar, expand for full shard grid
        card.innerHTML =
          '<div class="model-card-title">' +
            '<div class="model-card-name-row">' +
              chevronHtml +
              creatorIconHtml +
              '<span class="model-name" title="' + U.escapeHtml(m.id) + '">' + U.escapeHtml(name) + '</span>' +
              archTag + quantTag +
              compositeBadgeHtml +
            '</div>' +
            '<div class="model-card-controls">' +
              metaBtnHtml + gearHtml +
            '</div>' +
          '</div>' +
          availBarHtml +
          '<div class="model-card-shards">' +
            progressHtml + perShardDlHtml +
            '<div class="model-card-expanded">' +
              '<div class="mce-left">' +
                (detailBadgesHtml || '') +
                (healthBadgeHtml || '') +
                '<div class="mce-meta">' +
                  '<div class="mce-meta-row">' + footerMetaHtml + '</div>' +
                '</div>' +
                '<div class="mce-actions">' + actionHtml + removeHtml + '</div>' +
                '<div class="model-ticker" data-model-ticker="' + safeId + '" style="display:none"></div>' +
              '</div>' +
              '<div class="mce-right" data-shard-detail="' + safeId + '">' +
                '<div class="mce-right-head">' +
                  _buildCoverageRibbon(m, shards, safeId) +
                  _buildShardViewToggle() +
                '</div>' +
                '<div class="mce-right-body">' + _buildShardDetailBody(m, shards, safeId) + '</div>' +
              '</div>' +
            '</div>' +
          '</div>' +
          '<div class="model-card-footer">' +
            '<div class="model-card-meta">' + fileIndicators + '</div>' +
          '</div>' +
          '<div class="gguf-metadata-panel hidden" data-meta-panel="' + U.escapeHtml(m.id) + '"></div>';

        if (swarmBody) swarmBody.appendChild(card);

        // Restore per-model activity ticker from stored events (DOM only, don't re-log)
        if (_modelEvents[m.id] && _modelEvents[m.id].length > 0) {
          App.dashboard._renderModelTicker(m.id);
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
        card.innerHTML =
          '<div class="cloud-card-header' + (opts.headerClass ? ' ' + opts.headerClass : '') + '">' +
            '<span class="cloud-provider-name">' + (cardIconHtml ? cardIconHtml + ' ' : '') + U.escapeHtml(pLabel) + '</span>' +
            '<span style="display:flex;align-items:center;gap:8px">' +
              '<span class="badge ' + (opts.badgeClass || 'badge-cloud') + '">' + I18n.t('dashboard.cloud_models_count', { count: pModels.length }) + '</span>' +
              (opts.statusHtml || '<span class="cloud-status-ok">\u25cf ' + U.escapeHtml(I18n.t('dashboard.cloud_connected')) + '</span>') +
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
        setTimeout(function() { App.providerHealth.probe(sorted.slice(0, 20).map(function(cm) { return cm.id; })); }, 500);
        var filterEl = document.getElementById(filterId), sortEl = document.getElementById(sortId);
        var refreshRows = function() {
          var query = filterEl ? filterEl.value.toLowerCase().trim() : '';
          var sortBy = sortEl ? sortEl.value : 'popular';
          var filtered = query ? pModels.filter(function(cm) {
            return ((cm.name || '') + ' ' + cm.id + ' ' + (cm.meta && cm.meta.owned_by ? cm.meta.owned_by : '')).toLowerCase().indexOf(query) !== -1;
          }) : pModels;
          var s = sortCloudModels(filtered, sortBy);
          if (listContainer) renderRowsInto(listContainer, s);
          App.providerHealth.probe(s.slice(0, 20).map(function(cm) { return cm.id; }));
        };
        if (filterEl) { filterEl.addEventListener('input', refreshRows); filterEl.addEventListener('paste', function() { setTimeout(refreshRows, 0); }); }
        if (sortEl) sortEl.addEventListener('change', function() {
          refreshRows();
          if (sortEl.value === 'avail') App.providerHealth.probe(pModels.map(function(cm) { return cm.id; }).slice(0, 40));
        });
      }

      // --- Cloud provider models ---
      if (hasCloud) {
        var byProvider = {};
        apiModels.forEach(function(cm) {
          var p = cm.provider || 'cloud';
          if (!byProvider[p]) byProvider[p] = [];
          byProvider[p].push(cm);
        });

        var providerCount = Object.keys(byProvider).length;
        var cloudSection = document.createElement('details');
        cloudSection.className = 'models-section';
        cloudSection.open = true;
        var cloudMeta = I18n.t('dashboard.providers_count', { count: providerCount, models: apiModels.length });
        cloudSection.innerHTML = '<summary class="models-section-header">' +
          '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" aria-hidden="true" class="models-section-logo" style="flex-shrink:0"><path d="M18 10h-1.26A8 8 0 1 0 9 20h9a5 5 0 0 0 0-10z" fill="var(--accent)"/></svg>' +
          '<span class="models-section-title">' + U.escapeHtml(I18n.t('settings.cloud_providers')) + '</span>' +
          '<span class="models-section-count">' + cloudMeta + '</span>' +
          '</summary>';
        var cloudBody = document.createElement('div');
        cloudBody.className = 'models-section-body';
        cloudSection.appendChild(cloudBody);
        list.appendChild(cloudSection);

        Object.keys(byProvider).forEach(function(p) {
          renderProviderCard({ provider: p, models: byProvider[p], parentEl: cloudBody });
        });
        if (Object.keys(S.modelStatus).length > 0) App.providerHealth.updateModelBadges();
      }

      // --- Subscription models ---
      if (hasSubscription) {
        var bySubProvider = {};
        subscriptionModels.forEach(function(cm) {
          var p = cm.provider || 'subscription';
          if (!bySubProvider[p]) bySubProvider[p] = [];
          bySubProvider[p].push(cm);
        });

        var subProviderCount = Object.keys(bySubProvider).length;
        var subSection = document.createElement('details');
        subSection.className = 'models-section';
        subSection.open = true;
        var subMeta = I18n.t('dashboard.subscription_count', { count: subProviderCount, models: subscriptionModels.length });
        subSection.innerHTML = '<summary class="models-section-header">' +
          '<img src="' + (providerIconUrl('claude_subscription') || '') + '" width="16" height="16" alt="" aria-hidden="true" class="models-section-logo" style="flex-shrink:0">' +
          '<span class="models-section-title">' + U.escapeHtml(I18n.t('dashboard.subscription_title')) + '</span>' +
          '<span class="models-section-count">' + subMeta + '</span>' +
          '<span class="badge badge-subscription">' + U.escapeHtml(I18n.t('dashboard.subscription_badge')) + '</span>' +
          '</summary>';
        var subBody = document.createElement('div');
        subBody.className = 'models-section-body';
        subSection.appendChild(subBody);
        list.appendChild(subSection);

        Object.keys(bySubProvider).forEach(function(p) {
          renderProviderCard({
            provider: p, models: bySubProvider[p], parentEl: subBody,
            cardClass: 'subscription-model-card', headerClass: 'subscription-card-header',
            badgeClass: 'badge-subscription',
            statusHtml: '<span class="cloud-status-sub" id="sub-status-' + p + '">\u25cf ' + U.escapeHtml(I18n.t('dashboard.cloud_subscription')) + '</span>',
            noteText: I18n.t('dashboard.cloud_sub_note'),
            idPrefix: 'sub',
          });
        });

        // Fetch CLI status for subscription providers (dedup-coalesced across components)
        App.data.loadClaudeSubStatus().then(function(data) {
          if (!data || data.error) return;
          var statusEl = document.getElementById('sub-status-claude_subscription');
          if (!statusEl) return;
          var parts = [];
          if (data.authenticated) {
            parts.push('\u2713 ' + I18n.t('dashboard.sub_authenticated'));
            if (data.subscription_type) parts.push(data.subscription_type);
            if (data.cli_version) parts.push('v' + data.cli_version);
            statusEl.innerHTML = '<span style="color:var(--green)">\u25cf</span> ' + U.escapeHtml(parts.join(' \u00b7 '));
          } else {
            statusEl.innerHTML = '<span style="color:var(--red)">\u25cf</span> ' + U.escapeHtml(I18n.t('dashboard.sub_not_authenticated'));
            statusEl.style.color = 'var(--red)';
          }
        }).catch(function() {});
      }
    },

    updateShardsLive: function(acquisitions, shardRegistry, peerDownloads) {
      if (!acquisitions && !shardRegistry && !peerDownloads) return;

      if (acquisitions) {
        acquisitions.forEach(function(acq) {
          var modelId = acq.model_id;
          if (!modelId) return;
          var safeId = U.safeId(modelId);

          var shardDetails = acq.shard_details || [];
          var localCount = 0, peerCount = 0, dlCount = 0, peerDlCount = 0, queuedCount = 0, missingCount = 0;
          shardDetails.forEach(function(sd) {
            var cellId = safeId + '-' + sd.index;
            var cell = document.querySelector('[data-shard="' + cellId + '"]');
            if (!cell) return;

            var oldClass = cell.className.replace(/shard-cell\s*/, '').trim().split(/\s+/)[0] || 'missing';
            var newClass = 'missing';
            var label = '' + (sd.index + 1);
            var dlPct = sd.progress_pct || 0;

            if (sd.state === 'complete') { newClass = 'local'; localCount++; }
            else if (sd.state === 'verifying') {
              newClass = 'verifying'; dlCount++;
              label = '\u2713';
            } else if (sd.state === 'downloading') {
              newClass = 'downloading'; dlCount++;
              label = dlPct + '%';
            } else if (sd.state === 'pending') {
              newClass = 'queued'; queuedCount++;
              label = '\u2022';
            } else if (sd.state === 'failed') { newClass = 'missing'; missingCount++; }
            else { missingCount++; }

            if (oldClass !== newClass || cell.textContent !== label) {
              // Flash animation on state transition
              if (oldClass !== newClass) {
                cell.classList.add('shard-transitioning');
                setTimeout(function() { cell.classList.remove('shard-transitioning'); }, 1500);
                // Log per-model activity
                // Shard state transition logging is handled by backend activity_event
                // messages — no duplicate frontend logging needed here.
              }
              // Preserve lock, endpoint, and pinned classes
              var preserve = '';
              if (cell.classList.contains('locked')) preserve += ' locked';
              if (cell.classList.contains('shard-endpoint')) preserve += ' shard-endpoint';
              if (cell.classList.contains('shard-pinned')) preserve += ' shard-pinned';
              cell.className = 'shard-cell ' + newClass + preserve;

              // Set label text while preserving holder badge and endpoint tag
              var holderBadge = cell.querySelector('.shard-holders');
              var endpointTag = cell.querySelector('.shard-endpoint-tag');
              var lockIcon = cell.querySelector('.shard-lock-icon');
              // Clear text nodes only (preserves child elements)
              Array.from(cell.childNodes).forEach(function(n) {
                if (n.nodeType === 3) n.textContent = '';
              });
              cell.insertBefore(document.createTextNode(label), cell.firstChild);

              if (newClass === 'downloading' || newClass === 'peer-downloading') {
                cell.style.setProperty('--dl-pct', dlPct + '%');
              } else {
                cell.style.removeProperty('--dl-pct');
              }

              var title = I18n.t('shard.part_n', { n: sd.index + 1 });
              if (newClass === 'local') title += ' \u2014 ' + I18n.t('dashboard.shard_verified');
              else if (newClass === 'verifying') title += ' \u2014 ' + I18n.t('dashboard.shard_verifying');
              else if (newClass === 'downloading') title += ' \u2014 ' + I18n.t('shard.tooltip_downloading', { pct: dlPct });
              else if (newClass === 'queued') title += ' \u2014 ' + I18n.t('dashboard.shard_queued');
              else if (sd.state === 'failed') title += ' \u2014 ' + I18n.t('dashboard.shard_failed');
              else title += ' \u2014 ' + I18n.t('shard.tooltip_unavailable');
              cell.setAttribute('title', title);
              cell.setAttribute('aria-label', title);
            }
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

      // Update shard cells from shard registry changes
      if (shardRegistry) {
        Object.keys(shardRegistry).forEach(function(modelId) {
          var safeId = U.safeId(modelId);
          var shards = shardRegistry[modelId] || [];
          shards.forEach(function(s) {
            var cellId = safeId + '-' + s.index;
            var cell = document.querySelector('[data-shard="' + cellId + '"]');
            if (!cell) return;

            var current = cell.className;
            if (current.indexOf('downloading') >= 0) return;
            // Allow local→local+vram and local+vram→local transitions
            var alreadyLocal = current.indexOf('local') >= 0;
            var hasVram = current.indexOf('vram') >= 0;
            if (alreadyLocal && s.local && s.in_vram === hasVram) return;

            // Preserve lock/endpoint classes across state changes
            var preserve = '';
            if (cell.classList.contains('locked')) preserve += ' locked';
            if (cell.classList.contains('shard-endpoint')) preserve += ' shard-endpoint';
            if (cell.classList.contains('shard-pinned')) preserve += ' shard-pinned';

            if (s.local) {
              var wasLocal = alreadyLocal;
              if (!wasLocal) {
                cell.classList.add('shard-transitioning');
                setTimeout(function() { cell.classList.remove('shard-transitioning'); }, 1500);
              }
              var vramCls = s.in_vram ? 'local vram' : 'local';
              var vramLabel = s.in_vram ? I18n.t('dashboard.shard_active_vram', { mem: S._gpuInference ? 'VRAM' : 'RAM' }) : I18n.t('shard.tooltip_on_disk');
              cell.className = 'shard-cell ' + vramCls + preserve;
              // Preserve inner elements (holder badge, endpoint tag)
              Array.from(cell.childNodes).forEach(function(n) { if (n.nodeType === 3) n.textContent = ''; });
              cell.insertBefore(document.createTextNode('' + (s.index + 1)), cell.firstChild);
              cell.setAttribute('title', I18n.t('shard.part_n', { n: s.index + 1 }) + ' \u2014 ' + vramLabel);
              // Only log the first time a shard becomes local (not on vram toggle)
              // Shard state logging handled by backend activity_event
            } else if (s.holders > 0 && current.indexOf('peer') < 0) {
              cell.classList.add('shard-transitioning');
              setTimeout(function() { cell.classList.remove('shard-transitioning'); }, 1500);
              cell.className = 'shard-cell peer' + preserve;
              // Update or create holder badge
              var hBadge = cell.querySelector('.shard-holders');
              if (hBadge) { hBadge.textContent = s.holders; }
              else {
                hBadge = document.createElement('span');
                hBadge.className = 'shard-holders';
                hBadge.textContent = s.holders;
                cell.appendChild(hBadge);
              }
              cell.setAttribute('title', I18n.t('dashboard.shard_peer_available', { idx: s.index, holders: s.holders }));
              // Peer discovery logging handled by backend activity_event
            } else if (s.holders > 0) {
              // Update holder count on existing peer cells
              var hBadge2 = cell.querySelector('.shard-holders');
              if (hBadge2) hBadge2.textContent = s.holders;
            }
          });
        });
      }

      // Update shard cells with peer download progress
      if (peerDownloads && peerDownloads.length > 0) {
        peerDownloads.forEach(function(pd) {
          var safeId = U.safeId(pd.model_id);
          var cellId = safeId + '-' + pd.shard_index;
          var cell = document.querySelector('[data-shard="' + cellId + '"]');
          if (!cell) return;

          var current = cell.className;
          if (current.indexOf('local') >= 0 || current.indexOf(' downloading') >= 0) return;

          var pdPct = pd.progress_pct || 0;
          var wasPeerDl = current.indexOf('peer-downloading') >= 0;
          if (!wasPeerDl) {
            cell.classList.add('shard-transitioning');
            setTimeout(function() { cell.classList.remove('shard-transitioning'); }, 1500);
            App.dashboard._logModelEvent(pd.model_id, '\u{1F4E1}', I18n.t('dashboard.peer_downloading_log', { peer: pd.node_id.substring(0, 8), shard: pd.shard_index === App.MMPROJ_SHARD_INDEX ? 'mmproj' : pd.shard_index + 1 }), true);
          }
          var pdPreserve = '';
          if (cell.classList.contains('locked')) pdPreserve += ' locked';
          if (cell.classList.contains('shard-endpoint')) pdPreserve += ' shard-endpoint';
          if (cell.classList.contains('shard-pinned')) pdPreserve += ' shard-pinned';
          cell.className = 'shard-cell peer-downloading' + pdPreserve;
          cell.style.setProperty('--dl-pct', pdPct + '%');
          Array.from(cell.childNodes).forEach(function(n) { if (n.nodeType === 3) n.textContent = ''; });
          cell.insertBefore(document.createTextNode(pdPct + '%'), cell.firstChild);
          cell.setAttribute('title', I18n.t('dashboard.peer_downloading_title', { idx: pd.shard_index, peer: pd.node_id.substring(0, 8), pct: pdPct }));
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
        var lanBadge = p.is_lan_peer ? ' <span class="lan-badge">' + U.escapeHtml(I18n.t('dashboard.lan_badge')) + '</span>' : '';
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
