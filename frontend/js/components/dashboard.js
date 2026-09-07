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

  // Shard rendering helpers live in App.dashboardShards (dashboard-shards.js).
  // Alias to local names for tight call-site rewrites.
  var DS = App.dashboardShards;
  var MMPROJ_SHARD_INDEX = DS.MMPROJ_SHARD_INDEX;
  var _buildProgressBar    = DS.buildProgressBar;
  var _shardState          = DS.shardState;
  var _shardGlyph          = DS.shardGlyph;
  var _buildPieceBar       = DS.buildPieceBar;
  var _buildShardMatrix    = DS.buildShardMatrix;
  var _buildShardDetailBody = DS.buildShardDetailBody;
  var _buildShardViewToggle = DS.buildShardViewToggle;
  var _buildCoverageRibbon = DS.buildCoverageRibbon;

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
      if (state === 'peer' || state === 'missing' || state === 'gossip') destructive.push('<button data-shard-act="download">' + U.escapeHtml(I18n.t('shard.download')) + '</button>');
      if (shard.local) destructive.push('<button class="danger" data-shard-act="delete">' + U.escapeHtml(I18n.t('shard.row.action_delete')) + '</button>');

      var panelHtml = '<div class="shard-row-expanded-panel">' +
        '<div class="srep-section">' +
          '<div class="srep-section-label">' + U.escapeHtml(I18n.t('shard.row.holders_title')) + '</div>' +
          '<div class="srep-holders">' + holdersHtml + '</div>' +
        '</div>' +
        (shard.size_bytes ? '<div class="srep-section"><span class="srep-section-label">' + U.escapeHtml(I18n.t('shard.row.size_label')) + '</span> ' + U.formatBytes(shard.size_bytes) + '</div>' : '') +
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
    _setGauge: function(id, pct) {
      var el = document.getElementById(id);
      if (!el) return;
      var p = Math.max(0, Math.min(100, pct || 0));
      el.style.setProperty('--pct', p.toFixed(1));
      el.setAttribute('data-pct', Math.round(p));
      el.classList.remove('gauge-warn', 'gauge-crit');
      if (p > 90) el.classList.add('gauge-crit');
      else if (p > 70) el.classList.add('gauge-warn');
    },
    _updateContribution: function(pct, memKind) {
      var el = document.getElementById('contribution-pct');
      if (!el) return;
      var tier = pct < 5 ? 'idle' : pct < 25 ? 'minimal' : pct < 60 ? 'moderate' : 'maximum';
      var tierLabel = I18n.t('dashboard.contribution_tier_' + tier);
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

      App.data.loadPipelinePlan(modelId)
        .then(function(plan) {
          if (!plan || !plan.segments || plan.segments.length === 0) return;
          // Cache plan on the matrix so resize-driven redraws can reuse it
          // without re-fetching from the server.
          matrix._pipelinePlan = plan;
          // Retry on rAF until the table has real dimensions. On initial card
          // expand the fetch can resolve before layout settles; without this
          // the line stays invisible until the user toggles views.
          var attempts = 0;
          var tryDraw = function() {
            if (!matrix.isConnected) return;
            var tbl = matrix.querySelector('table');
            if (tbl && tbl.clientWidth > 0 && tbl.clientHeight > 0) {
              App.dashboard._drawPipelinePath(matrix);
              return;
            }
            if (++attempts < 30) requestAnimationFrame(tryDraw);
          };
          tryDraw();
          // Redraw on later layout changes (peers-list expand, window resize).
          if (!matrix._pipelineRO && typeof ResizeObserver !== 'undefined') {
            var ro = new ResizeObserver(function() { App.dashboard._drawPipelinePath(matrix); });
            ro.observe(table);
            matrix._pipelineRO = ro;
          }
          // Also redraw when the matrix itself becomes visible (display:none →
          // block on view switch). ResizeObserver misses this transition on
          // some browsers when the node is inserted already hidden.
          if (!matrix._pipelineIO && typeof IntersectionObserver !== 'undefined') {
            var io = new IntersectionObserver(function(entries) {
              entries.forEach(function(e) {
                if (e.isIntersecting) App.dashboard._drawPipelinePath(matrix);
              });
            });
            io.observe(matrix);
            matrix._pipelineIO = io;
          }
        })
        .catch(function() { /* quiet: plan unavailable (no peers etc.) */ });
    },

    _drawPipelinePath: function(matrix) {
      if (!matrix) return;
      var plan = matrix._pipelinePlan;
      var table = matrix.querySelector('table');
      var svg = matrix.querySelector('.shard-matrix-path');
      if (!plan || !table || !svg) return;
      if (table.clientWidth === 0 || table.clientHeight === 0) return;
      svg.innerHTML = '';
      // Reset row highlights from any previous draw.
      matrix.querySelectorAll('.planned-cell').forEach(function(el) {
        el.classList.remove('planned-cell');
        el.removeAttribute('data-plan-order');
      });
      matrix.querySelectorAll('tbody tr').forEach(function(tr) {
        tr.classList.remove('planned-row');
        tr.classList.add('unplanned-row');
      });
      matrix.setAttribute('data-has-plan', '1');

      var localId = plan.local_node_id;
      var U = App.utils;
      var points = [];
      var tblRect = table.getBoundingClientRect();
      plan.segments.forEach(function(seg, i) {
            var peerId = seg.node_id;
            var row = peerId === localId
              ? matrix.querySelector('tr.srm-row-self')
              : matrix.querySelector('tr.srm-row-peer[data-peer-id="' + U.cssSafeAttr(peerId) + '"]');
            if (!row) return;
            row.classList.remove('unplanned-row');
            row.classList.add('planned-row');
            // Mark every cell this segment covers (a peer may serve multiple
            // contiguous shards as one segment).
            var indices = (seg.shard_indices && seg.shard_indices.length)
              ? seg.shard_indices.slice().sort(function(a, b) { return a - b; })
              : [seg.shard_index];
            var segCells = [];
            indices.forEach(function(idx) {
              var td = row.querySelector('td[data-shard-col="' + idx + '"]');
              if (!td) return;
              td.classList.add('planned-cell');
              segCells.push(td);
            });
            if (segCells.length === 0) return;
            // Route the polyline along the chosen cells: enter at the first,
            // traverse to the last. The hop label sits on the first cell.
            segCells[0].setAttribute('data-plan-order', String(i + 1));
            segCells.forEach(function(td, k) {
              var r = td.getBoundingClientRect();
              points.push({
                x: (r.left + r.width / 2) - tblRect.left,
                y: (r.top + r.height / 2) - tblRect.top,
                local: seg.is_local,
                anchor: k === 0,
                label: k === 0 ? String(i + 1) : '',
              });
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
          // Draw a dot on every cell along the path so the traversal is
          // visible end-to-end. Number only the segment anchors (1, 2, 3...).
          points.forEach(function(p) {
            var c = document.createElementNS(ns, 'circle');
            c.setAttribute('cx', p.x);
            c.setAttribute('cy', p.y);
            c.setAttribute('r', p.anchor ? '6' : '3');
            c.setAttribute('class', 'shard-matrix-path-dot' + (p.local ? ' local' : '') + (p.anchor ? ' anchor' : ''));
            svg.appendChild(c);
            if (p.label) {
              var t = document.createElementNS(ns, 'text');
              t.setAttribute('x', p.x);
              t.setAttribute('y', p.y + 3);
              t.setAttribute('class', 'shard-matrix-path-label');
              t.textContent = p.label;
              svg.appendChild(t);
            }
          });
    },
    _tickerSig: {},
    _renderModelTicker: function(modelId) {
      var actEvents = _modelEvents[modelId] || [];
      var netEvents = _modelNetEvents[modelId] || [];
      if (actEvents.length === 0 && netEvents.length === 0) return;

      var safeId = U.safeId(modelId);
      var ticker = document.querySelector('[data-model-ticker="' + safeId + '"]');
      if (!ticker) return;

      // Skip the innerHTML rebuild when nothing has changed. Each ticker is
      // re-rendered on every activity event for that model; during download
      // bursts that's many events per second per model. The signature
      // collapses to top event ts + length on each side.
      var actTop = actEvents.length ? actEvents[0].ts : 0;
      var netTop = netEvents.length ? netEvents[0].ts : 0;
      var sig = actTop + ':' + actEvents.length + '|' + netTop + ':' + netEvents.length;
      if (App.dashboard._tickerSig[modelId] === sig) return;
      App.dashboard._tickerSig[modelId] = sig;

      function _tickerTime(ts) {
        var d = new Date(ts);
        return ('0' + d.getHours()).slice(-2) + ':' + ('0' + d.getMinutes()).slice(-2) + ':' + ('0' + d.getSeconds()).slice(-2);
      }
      function _renderColumn(events, emptyText) {
        if (events.length === 0) return '<div class="text-muted text-2xs py-1">' + U.escapeHtml(emptyText) + '</div>';
        var latest = events[0];
        var html = '<div class="model-ticker-latest"><span class="model-ticker-icon">' + U.escapeHtml(latest.icon) + '</span>' +
          '<span class="model-ticker-text">' + U.escapeHtml(latest.text) + '</span>' +
          '<span class="model-ticker-time" data-ts="' + latest.ts + '">' + U.timeAgo(latest.ts) + '</span></div>';
        if (events.length > 1) {
          html += '<div class="model-ticker-history">';
          events.slice(1, 6).forEach(function(e) {
            html += '<div class="model-ticker-row"><span>' + U.escapeHtml(e.icon) + ' ' + U.escapeHtml(e.text) + '</span><span class="model-ticker-time" data-ts="' + e.ts + '">' + _tickerTime(e.ts) + ' ' + U.timeAgo(e.ts) + '</span></div>';
          });
          html += '</div>';
        }
        return html;
      }

      ticker.innerHTML =
        '<div class="model-ticker-split">' +
          '<div class="model-ticker-col"><div class="model-ticker-col-label">' + U.escapeHtml(I18n.t('activity.label_activity')) + '</div>' + _renderColumn(actEvents, I18n.t('activity.none')) + '</div>' +
          '<div class="model-ticker-col"><div class="model-ticker-col-label">' + U.escapeHtml(I18n.t('activity.label_network')) + '</div>' + _renderColumn(netEvents, I18n.t('activity.none_network')) + '</div>' +
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

    // The hardware panel, rendered from whichever payload carried it.
    //
    // Extracted so the 2-second live tick can render it too. It used to
    // live inside `updateFull`, which only runs on an explicit REST fetch,
    // so every other figure on the dashboard updated live and this panel
    // sat still until something happened to refetch (report #016).
    _renderHardware: function(hw) {
      if (!hw) return;
        S._gpuInference = !!hw.gpu_inference;
        if (App.settings && App.settings.renderHwModeNote) {
          App.settings.renderHwModeNote(document.getElementById('hw-mode-note'), S._gpuInference);
        }
        var gpuEl = document.getElementById('node-gpu');
        var gpuBadge = document.getElementById('node-gpu-badge');
        if (hw.gpu_name) {
          gpuEl.textContent = hw.gpu_name;
          if (gpuBadge) {
            if (hw.gpu_inference) {
              var backendLabel = hw.inference_backend || 'GPU';
              gpuBadge.textContent = I18n.t('hw.gpu_mode_label', { backend: backendLabel });
              gpuBadge.className = 'node-mode-badge node-mode-badge-interactive node-mode-gpu';
              gpuBadge.removeAttribute('title');
            } else {
              gpuBadge.textContent = I18n.t('hw.mode_cpu');
              gpuBadge.className = 'node-mode-badge node-mode-badge-interactive node-mode-cpu';
              gpuBadge.removeAttribute('title');
            }
          }
          if (hw.gpu_vram_mb) {
            var vramUsed = hw.gpu_vram_used_mb || 0;
            var vramTotal = hw.gpu_vram_mb;
            var vramEl = document.getElementById('node-vram');

            var vramLabel = document.getElementById('vram-label');
            // The VRAM gauge always shows the card, whether or not inference is
            // bound to it. What the card is doing is a separate question from
            // what this node is contributing, which is decided once below.
            App.dashboard._setGauge('vram-gauge', vramTotal > 0 ? (vramUsed / vramTotal * 100) : 0);
            if (hw.gpu_inference) {
              if (vramLabel) vramLabel.textContent = I18n.t('hw.vram');
              // ALWAYS show live VRAM (hw.gpu_vram_used_mb, read from
              // nvidia-smi by the backend) as the primary number and as the
              // gauge input.
              //
              // This used to display the SUM of every loaded model's
              // estimated_vram_mb instead, whenever any model was "loaded".
              // That estimate is derived from model size, not from the device,
              // and it only got a clarifying tooltip when real usage EXCEEDED
              // it. In the other direction it was silently wrong: the
              // 2026-07-21 report had 5 loaded models estimating ~5.3 GB
              // rendering as "5.3 GB / 5.7 GB — 93% VRAM" on a red gauge while
              // nvidia-smi and the API agreed the real figure was ~1 GB (17%).
              // An idle machine looked like it was about to OOM, and the
              // reporter spent time chasing it as a sixth bug.
              var activeVramMb = 0;
              if (App.data.cache.models && App.data.cache.models.length) {
                App.data.cache.models.forEach(function(m) {
                  if (m.status === 'loaded' && m.estimated_vram_mb) activeVramMb += m.estimated_vram_mb;
                });
              }
              vramEl.textContent = U.formatMB(vramUsed) + ' / ' + U.formatMB(vramTotal);
              // The committed estimate is still worth surfacing — it's what
              // auto-manage budgets against — but only ever as clearly
              // labelled secondary context, never as the headline figure.
              if (activeVramMb > 0) {
                vramEl.title = I18n.t('hw.vram_live_tip', {
                  used: U.formatMB(vramUsed),
                  committed: U.formatMB(activeVramMb)
                });
              } else {
                vramEl.title = I18n.t('hw.vram_live_only_tip', { used: U.formatMB(vramUsed) });
              }
            } else {
              // The card is present but idle — driver baseline only.
              if (vramLabel) vramLabel.textContent = I18n.t('hw.vram_idle');
              vramEl.textContent = U.formatMB(vramUsed) + ' / ' + U.formatMB(vramTotal);
              vramEl.title = I18n.t('hw.vram_idle_tip');
            }
          }
        } else {
          // No card was DETECTED — which is two different facts, and saying
          // the wrong one misinforms the person who can least check it. With
          // no GPU backend compiled into this build there is nothing that
          // could have detected a card, so "None" would be a claim about
          // hardware made by a binary that never looked. The Mac artifact
          // ships neither backend, so every Apple machine read "None / CPU
          // only" while its GPU sat there, real and simply unaddressed
          // (report #019).
          var backendCanSeeAGpu = hw.gpu_backend_in_build !== false;
          gpuEl.textContent = backendCanSeeAGpu
            ? I18n.t('hw.none')
            : I18n.t('hw.no_gpu_backend');
          gpuEl.title = backendCanSeeAGpu ? '' : I18n.t('hw.no_gpu_backend_tip');
          if (gpuBadge) {
            gpuBadge.textContent = I18n.t('hw.mode_cpu_only');
            gpuBadge.className = 'node-mode-badge node-mode-badge-interactive node-mode-cpu';
            gpuBadge.removeAttribute('title');
          }
          // On unified memory there is no second pool to report. An em dash
          // reads as "unknown"; the truth is that the figure in the other
          // card already covers this one.
          var vramCell = document.getElementById('node-vram');
          if (hw.unified_memory) {
            vramCell.textContent = I18n.t('hw.unified_memory');
            vramCell.title = I18n.t('hw.unified_memory_tip');
          } else {
            vramCell.textContent = '\u2014';
            vramCell.title = '';
          }
        }

        // ── Your contribution ─────────────────────────────────────────────
        // ONE decision, taken after the readouts above rather than inside
        // them. It used to be made in two of the three display branches, and
        // the branch it was missing from is the one a machine with no
        // graphics card takes — so a processor-only node, the hardware this
        // project explicitly supports, showed a flat 0% however hard it was
        // working, and `_updateContribution` was never called at all
        // (report #011).
        //
        // The bar tracks whichever memory the work actually uses: the card
        // when inference is bound to it, otherwise everything SwarmLLM has
        // resident — daemon plus its model workers — against total RAM.
        var contribPct = 0;
        var contribKind = 'ram';
        if (hw.gpu_inference && hw.gpu_vram_mb) {
          contribKind = 'vram';
          contribPct = (hw.gpu_vram_used_mb || 0) / hw.gpu_vram_mb * 100;
        } else if (hw.total_ram_mb > 0) {
          contribPct = (hw.process_rss_mb || 0) / hw.total_ram_mb * 100;
        }
        contribPct = Math.max(0, Math.min(100, contribPct));
        var contribBar = document.getElementById('vram-bar');
        if (contribBar) {
          contribBar.style.width = contribPct.toFixed(1) + '%';
          contribBar.className = U.resourceBarClass(contribPct, 'cyan');
        }
        App.dashboard._updateContribution(contribPct, contribKind);
        document.getElementById('node-cpu').textContent = hw.cpu_name ? hw.cpu_name + ' ' + I18n.t('hw.cores', { cores: hw.cpu_cores }) : I18n.t('hw.unknown_cpu');

        if (hw.total_ram_mb) {
          document.getElementById('ram-total').textContent = '/ ' + U.formatMB(hw.total_ram_mb);
          // Show per-process RSS (this node's actual memory) rather than system-wide
          var processRss = hw.process_rss_mb || 0;
          var ramUsed = processRss > 0 ? processRss : (hw.used_ram_mb || 0);
          var ramEl = document.getElementById('ram-used');
          ramEl.textContent = U.formatMB(ramUsed);
          if (processRss > 0) {
            // Name the two parts. This figure is the daemon PLUS every model
            // worker; until report #011 it was the daemon alone, so anyone who
            // remembers the old number needs to see where the rest came from.
            var ramLines = [U.formatMB(processRss)];
            if (hw.worker_count) {
              ramLines.push(I18n.t('hw.ram_split', {
                count: hw.worker_count,
                daemon: U.formatMB(hw.daemon_rss_mb || 0),
                workers: U.formatMB(hw.worker_rss_mb || 0)
              }));
            }
            ramLines.push(I18n.t(S._gpuInference ? 'hw.ram_tip_gpu' : 'hw.ram_tip_cpu'));
            ramLines.push(U.formatMB(hw.used_ram_mb || 0) + ' / ' + U.formatMB(hw.total_ram_mb));
            ramEl.title = ramLines.join('\n\n');
          }
          var ramPct = hw.total_ram_mb > 0 ? (ramUsed / hw.total_ram_mb * 100) : 0;
          document.getElementById('ram-bar').style.width = ramPct.toFixed(1) + '%';
          document.getElementById('ram-bar').className = U.resourceBarClass(ramPct, 'green');
          App.dashboard._setGauge('ram-gauge', ramPct);
        }
        if (hw.total_disk_mb) {
          document.getElementById('disk-total').textContent = '/ ' + U.formatMB(hw.total_disk_mb);
          var diskUsed = hw.used_disk_mb || 0;
          document.getElementById('disk-used').textContent = U.formatMB(diskUsed);
          var diskPct = hw.total_disk_mb > 0 ? (diskUsed / hw.total_disk_mb * 100) : 0;
          var diskBar = document.getElementById('disk-bar');
          diskBar.style.width = diskPct.toFixed(1) + '%';
          diskBar.className = U.resourceBarClass(diskPct, 'accent');
          App.dashboard._setGauge('disk-gauge', diskPct);
        }
    },
    updateFull: function(data) {
      if (data.version) {
        var vEl = document.getElementById('app-version');
        if (vEl) {
          vEl.textContent = 'v' + data.version;
          vEl.removeAttribute('hidden');
        }
      }
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
              successLabel: I18n.t('actions.copied'),
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
      }

      App.dashboard.updateStats(data);

      App.dashboard._renderHardware(data.hardware);

      if (data.hosted_shards !== undefined) document.getElementById('hosted-shards').textContent = data.hosted_shards;
    },

    updateStats: function(data) {
      // The live 2-second tick carries the hardware panel now (report #016).
      // It used to be rendered only by `updateFull`, which runs on an explicit
      // fetch, so RAM and the graphics gauge sat still while every other figure
      // on the dashboard moved. The backend kept it off the tick because
      // measuring it cost 182 ms — that was fixed on 2026-09-06 and is now
      // 0.43 ms, so the reason no longer applies.
      if (data && data.hardware) App.dashboard._renderHardware(data.hardware);
      // Node version in the header, next to the logo. Guarded so the every-2s
      // WS tick doesn't rewrite the DOM when the (static) version is unchanged.
      if (data.version) {
        var vEl = document.getElementById('app-version');
        if (vEl && vEl.getAttribute('data-v') !== data.version) {
          vEl.textContent = 'v' + data.version;
          vEl.setAttribute('data-v', data.version);
          vEl.removeAttribute('hidden');
        }
      }
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
      // The credit balance is deliberately not rendered. Every element this
      // block used to write to is gone from the dashboard: the figure is
      // self-minted and reconciled with nobody, so presenting it as earnings
      // was a claim the product could not stand behind
      // (`docs/CREDITS_DESIGN.md`). The daemon still reports `data.credits`
      // and still keeps the books; nothing shows them to the user.
      if (data.requests_served !== undefined) _trackStat('served', data.requests_served, 'stat-served');
      if (data.requests_made !== undefined) _trackStat('requests', data.requests_made, 'stat-requests-made');
      if (data.forwards_served !== undefined) _trackStat('forwards', data.forwards_served, 'stat-forwards');
      if (data.active_requests !== undefined) _trackStat('active', data.active_requests, 'stat-active');

      // Capacity facts and connection state are now rendered together
      // by App.networkStatus (replaced the R110 swarm-capacity-banner +
      // the older mode-indicator strip with a single status panel).
      App.networkStatus.update(data, S._cachedProviderData);

      if (typeof NeuralBg !== 'undefined') NeuralBg.updateState(data);
    },


    renderModels: function(models, cloudModels) {
      // models cached in App.data.cache.models
      var list = document.getElementById('models-list');
      var empty = document.getElementById('models-empty');
      var loading = document.getElementById('models-loading');
      if (loading) loading.remove();

      // Keep the empty-state copy honest about connection state. Once we have
      // any swarm peer, "Connecting to the network…" is wrong — we're already
      // connected, there just aren't any shared models yet. Update data-i18n
      // too so a later language switch keeps the right variant.
      (function () {
        var txt = document.getElementById('models-empty-text');
        if (!txt) return;
        var st = (App.data && App.data.cache && App.data.cache.stats) || null;
        var connected = !!(st && (st.peers || 0) > 0);
        var key = connected ? 'models.empty_connected' : 'models.empty_state';
        txt.setAttribute('data-i18n', key);
        txt.textContent = I18n.t(key);
      })();

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

      // Disconnect any per-card observers (Resize/Intersection) before
      // wiping the DOM. Otherwise the observers keep firing on detached
      // nodes and accumulate across each models_changed re-render —
      // observable as a slow leak on a long-running dashboard tab.
      function _disconnectMatrixObservers(root) {
        if (!root) return;
        var matrices = root.querySelectorAll('[data-shard-matrix]');
        for (var i = 0; i < matrices.length; i++) {
          var m = matrices[i];
          if (m._pipelineRO) { try { m._pipelineRO.disconnect(); } catch (e) {} m._pipelineRO = null; }
          if (m._pipelineIO) { try { m._pipelineIO.disconnect(); } catch (e) {} m._pipelineIO = null; }
        }
      }

      if ((!models || models.length === 0) && !hasCloud && !hasSubscription) {
        _disconnectMatrixObservers(list);
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
        _disconnectMatrixObservers(list);
        list.innerHTML = '';
        empty.style.display = '';
        var _sb2 = document.getElementById('models-stats-bar');
        if (_sb2) _sb2.style.display = 'none';
        return;
      }

      empty.style.display = 'none';
      _disconnectMatrixObservers(list);
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
            '<option value="status"' + (swarmSort === 'status' ? ' selected' : '') + '>' + U.escapeHtml(I18n.t('dashboard.section_status')) + '</option>' +
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
          // The setting is ON but this node cannot satisfy it, so every request
          // for this model fails at pipeline assembly. Without its own state it
          // fell through to the amber "unprotected" branch below, which told the
          // user their prompts were unprotected when the truth was that nothing
          // worked at all — and offered no way to switch the setting back off.
          var encBlocked = !!m.encrypted_pipeline_blocked;
          // Blocked must stay toggleable: turning the setting OFF is the fix,
          // and disabling has no shard precondition (only enabling does).
          var canToggle = (hasFirst && hasLast) || encBlocked;
          var encState; // { stateMod, badgeCls, icon, label, detail, tip, action }
          if (encBlocked) {
            var blockedParts = [];
            if (!hasFirst) blockedParts.push(I18n.t('dashboard.enc_missing_first'));
            if (!hasLast)  blockedParts.push(I18n.t('dashboard.enc_missing_last', { n: shardCount }));
            var blockedMissing = blockedParts.length === 2
              ? I18n.t('enc.missing_both')
              : (blockedParts.length === 1 ? I18n.t('enc.missing_the', { which: blockedParts[0] }) : '');
            encState = {
              stateMod: 'mce-section-state-red', badgeCls: 'cb-incomplete',
              icon: '⚠', label: I18n.t('enc.blocked'),
              detail: I18n.t('enc.blocked_detail', { missing: blockedMissing }),
              tip: I18n.t('enc.blocked_tip'),
              action: I18n.t('enc.disable')
            };
          } else if (encActive) {
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
              detail: I18n.t('enc.ready_detail') + ' ' + I18n.t('enc.cost_detail'),
              tip: I18n.t('enc.ready_tip'),
              action: I18n.t('enc.enable_privacy'),
              recommended: true
            };
          } else {
            var missingParts2 = [];
            if (!hasFirst) missingParts2.push(I18n.t('dashboard.enc_missing_first'));
            if (!hasLast)  missingParts2.push(I18n.t('dashboard.enc_missing_last', { n: shardCount }));
            var missingText = missingParts2.length === 2
              ? I18n.t('enc.missing_both')
              : (missingParts2.length === 1 ? I18n.t('enc.missing_the', { which: missingParts2[0] }) : '');
            encState = {
              stateMod: 'mce-section-state-amber', badgeCls: 'cb-fragile',
              icon: '\uD83D\uDD13', label: I18n.t('enc.unavailable'),
              detail: I18n.t('enc.unprotected_detail', { missing: missingText }) + ' ' + I18n.t('enc.cost_detail'),
              tip: I18n.t('enc.unprotected_tip'),
              // Previously a dead end: it told you which pieces you lacked and
              // left you to find and download them yourself.
              action: '',
              fetchMissing: true
            };
          }
          var toggleAttrs = canToggle
            ? ' data-enc-toggle="' + U.escapeHtml(m.id) + '" data-enc-ready="1" role="switch" aria-checked="' + (encActive ? 'true' : 'false') + '"'
            : '';
          var toggleCls = canToggle ? ' mce-section-toggleable' : '';
          if (encState.fetchMissing) {
            encState.action = '';
          }
          var fetchHtml2 = encState.fetchMissing
            ? '<button class="btn btn-xs enc-banner-btn enc-banner-btn-enable" data-enc-fetch="' + U.escapeHtml(m.id) + '" title="' + U.escapeHtml(I18n.t('enc.cost_detail')) + '">' +
              U.escapeHtml(I18n.t('enc.fetch_missing')) +
              ' <span class="enc-recommended-badge">' + U.escapeHtml(I18n.t('enc.recommended')) + '</span></button>'
            : '';
          var actionHtml2 = canToggle && encState.action
            ? '<span class="mce-section-action">' + U.escapeHtml(encState.action) +
              (encState.recommended ? ' <span class="enc-recommended-badge">' + U.escapeHtml(I18n.t('enc.recommended')) + '</span>' : '') + '</span>'
            : '';
          privacySectionHtml =
            '<div class="mce-section mce-section-privacy ' + encState.stateMod + toggleCls + '"' + toggleAttrs + ' title="' + U.escapeHtml(encState.tip) + '">' +
              '<div class="mce-section-header">' +
                '<div class="mce-section-title">' + U.escapeHtml(I18n.t('dashboard.section_privacy')) + '</div>' +
                actionHtml2 + fetchHtml2 +
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

        // WHY THIS MODEL IS NOT ON THE GRAPHICS CARD.
        //
        // Rendered only when it is not, and only when this node holds shards —
        // a model served entirely by peers has no local placement to explain.
        // The backend already distinguishes the three causes
        // (`cpu_placement_reason`); before this the only place that answer
        // existed was one log line written at the moment the worker spawned,
        // which is no use to someone looking at a dashboard hours later. A
        // tester spent a session on exactly this question (2026-08-10).
        var placementSectionHtml = '';
        var cpuReason = m.cpu_placement_reason;
        // A HYBRID split: part of the model on the card, the rest on the
        // processor. Reported by the daemon from the worker it actually spawned
        // (`gpu_layers_on_card`); until this existed the only trace of the
        // split was one log line at spawn, and the card showed `fits_on_gpu:
        // true` for a model running 13 of its 28 layers on the card — so a
        // user could not tell why it was slower than the number promised.
        if (!cpuReason && hostedShards > 0 && typeof m.gpu_layers_on_card === 'number' && m.num_layers > 0) {
          placementSectionHtml =
            '<div class="mce-section mce-section-placement mce-section-state-amber" title="' + U.escapeHtml(I18n.t('placement.tip')) + '">' +
              '<div class="mce-section-header">' +
                '<div class="mce-section-title">' + U.escapeHtml(I18n.t('placement.section')) + '</div>' +
              '</div>' +
              '<div class="mce-section-body">' +
                '<span class="composite-badge cb-fragile">' +
                  '<span class="mce-section-icon">🖥</span>' +
                  U.escapeHtml(I18n.t('placement.hybrid', { on: m.gpu_layers_on_card, total: m.num_layers })) +
                '</span>' +
                '<div class="mce-section-detail">' + U.escapeHtml(I18n.t('placement.hybrid_detail')) + '</div>' +
              '</div>' +
            '</div>';
        }
        if (cpuReason && hostedShards > 0) {
          // `not_enough_vram` is the recoverable one and reads as amber; the
          // other two are settled facts about this machine until something
          // changes, so they are plain.
          var reasonKeys = {
            not_enough_vram:            { key: 'placement.not_enough_vram',  mod: 'mce-section-state-amber', badge: 'cb-fragile' },
            configured_cpu_only:        { key: 'placement.configured',       mod: '',                        badge: 'cb-ready' },
            gpu_too_old_for_this_build: { key: 'placement.gpu_too_old',      mod: 'mce-section-state-amber', badge: 'cb-fragile' }
          };
          var r = reasonKeys[cpuReason];
          if (r) {
            placementSectionHtml =
              '<div class="mce-section mce-section-placement ' + r.mod + '" title="' + U.escapeHtml(I18n.t('placement.tip')) + '">' +
                '<div class="mce-section-header">' +
                  '<div class="mce-section-title">' + U.escapeHtml(I18n.t('placement.section')) + '</div>' +
                '</div>' +
                '<div class="mce-section-body">' +
                  '<span class="composite-badge ' + r.badge + '">' +
                    '<span class="mce-section-icon">🖥</span>' +
                    U.escapeHtml(I18n.t('placement.on_cpu')) +
                  '</span>' +
                  '<div class="mce-section-detail">' + U.escapeHtml(I18n.t(r.key)) + '</div>' +
                '</div>' +
              '</div>';
          }
        }

        // YOUR own download progress is rendered exclusively by the global
        // Downloads panel (see frontend/js/components/downloads.js). The model
        // card never duplicates it — peer_downloads dots in the matrix view
        // already convey "swarm replication is active" via gossip from other
        // nodes.

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
        configRows.push(['dashboard.info_mode', S._gpuInference ? I18n.t('dashboard.gpu_label') : I18n.t('dashboard.cpu_label')]);
        // VRAM fit — only when GPU mode; in CPU mode the Mode row already conveys this.
        if (m.estimated_vram_mb && S._gpuInference) {
          // `fits_on_gpu` is the DAEMON's own verdict, judged against the
          // configured GPU budget. Prefer it: this panel used to re-derive the
          // answer from the card's TOTAL VRAM, which is a different and larger
          // number than admission uses, so the dashboard could show a
          // comfortable fit for a model the daemon had refused. Fall back to
          // the ratio only when the daemon cannot answer (no budget set, or the
          // model is held only by peers so there is no local geometry to read).
          var budget = m.gpu_budget_mb || (App.data.cache.stats && App.data.cache.stats.hardware && App.data.cache.stats.hardware.gpu_vram_mb) || 0;
          var fitClass = 'fit-no', fitLabel = U.formatMB(m.estimated_vram_mb);
          if (m.fits_on_gpu === true) { fitClass = 'fit-yes'; fitLabel = '\u2713 ' + fitLabel; }
          else if (m.fits_on_gpu === false) { fitClass = 'fit-no'; fitLabel = '\u2717 ' + fitLabel; }
          else if (budget > 0) {
            var ratio = m.estimated_vram_mb / budget;
            if (ratio <= 0.85) { fitClass = 'fit-yes'; fitLabel = '\u2713 ' + fitLabel; }
            else if (ratio <= 1.05) { fitClass = 'fit-tight'; fitLabel = '\u2248 ' + fitLabel; }
            else { fitClass = 'fit-no'; fitLabel = '\u2717 ' + fitLabel; }
          }
          configRows.push(['hw.vram', '<span class="vram-fit ' + fitClass + '" title="' + U.escapeHtml(I18n.t('dashboard.vram_fit_tip', { est: U.formatMB(m.estimated_vram_mb), total: budget > 0 ? U.formatMB(budget) : '?' })) + '">' + fitLabel + '</span>']);
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
                // Mark pinned test models so they are not mistaken for a
                // recommendation — they exist to make speed results
                // comparable, not because they are the best choice to chat to.
                ((App.referenceModels && App.referenceModels.isReference(m.id))
                  ? '<span class="badge badge-purple" title="' +
                      U.escapeHtml(I18n.t('reference.badge_hint')) + '">' +
                      U.escapeHtml(I18n.t('reference.badge')) + '</span>'
                  : '') +
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
                // PLACEMENT — only present when this node runs the model on its
                // processor, and says which of the three reasons it is.
                placementSectionHtml +
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
          ? '<span class="ph-tag tag-sub" title="' + U.escapeHtml(I18n.t('dashboard.cloud_sub_note')) + '">' + U.escapeHtml(I18n.t('dashboard.chip_subscription')) + '</span>'
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
              statusHtml: '<span class="badge badge-claude" id="sub-status-' + p + '">' + U.escapeHtml(I18n.t('dashboard.chip_subscription')) + '</span>',
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

      // Build a single rowId -> element map up front. The previous code
      // ran a fresh document.querySelector('[data-shard-row="..."]') per
      // shard per tick; with N models × M shards that's N×M tree scans
      // every 2 seconds. One querySelectorAll + map lookup is N×1 + O(N×M)
      // hash hits.
      var rowMap = {};
      var rows = document.querySelectorAll('[data-shard-row]');
      for (var ri = 0; ri < rows.length; ri++) {
        rowMap[rows[ri].getAttribute('data-shard-row')] = rows[ri];
      }
      function _findRow(rowId) {
        return rowMap[U.cssSafeAttr(rowId)] || null;
      }

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
            var row = _findRow(rowId);
            if (!row) return;

            var dlPct = sd.progress_pct || 0;
            var newState = 'missing';
            var statusText = I18n.t('shard.row.missing_label');
            if (sd.state === 'complete') { newState = 'disk'; statusText = I18n.t('dashboard.disk_label'); }
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
              // R137: unify speed separator with the active-update branch
              // above (line ~1706) which uses U+00B7 middle-dot. Em-dash
              // here was inconsistent \u2014 flagged as deferred in R123 sweep.
              var rightText2 = U.formatDlProgress(dlBytes2, acq.total_bytes, pct2) + (speed2 > 0 ? ' \u00b7 ' + U.formatSpeed(speed2) : '');
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

        });
      }

      // Patch shard rows from shardRegistry (peer availability snapshot).
      // The backend stats cache ships the full registry on every 2s tick;
      // skip the DOM patch loop when the registry hasn't changed since the
      // previous tick so we don't burn the compositor on a static swarm.
      if (shardRegistry) {
        var regStr = JSON.stringify(shardRegistry);
        if (regStr === self._lastShardRegistryStr) {
          // Unchanged — skip the patch loop. Peer downloads block below still
          // runs to reflect in-flight state that comes from peer_downloads
          // (a separate field on the stats payload).
          shardRegistry = null;
        } else {
          self._lastShardRegistryStr = regStr;
        }
      }
      if (shardRegistry) {
        Object.keys(shardRegistry).forEach(function(modelId) {
          var safeId = U.safeId(modelId);
          var shards = shardRegistry[modelId] || [];
          shards.forEach(function(s) {
            var rowId = safeId + '-' + s.index;
            var row = _findRow(rowId);
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
            else if (s.local) { newState = 'disk'; statusText = I18n.t('dashboard.disk_label'); }
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

      // Skip the full <table> rebuild when none of the rendered fields
      // changed. peer_list bursts on swarm churn (multiple per second) —
      // each rebuild destroys ~20 rows and recreates them from a string.
      // The signature covers every field the row template uses; sort
      // direction is included so toggle clicks still trigger a rerender.
      var renderSig = (peers || []).map(function(p) {
        return (p.node_id || '') + '|' + (p.healthy ? 1 : 0) + '|' +
          (p.latency_ms || 0) + '|' + (p.hosted_shards || 0) + '|' +
          (p.trust_score || 0) + '|' + (p.is_lan_peer ? 1 : 0) + '|' +
          (p.is_pool_member ? 1 : 0) + '|' +
          (p.nickname || '') + '|' + (p.gpu || '') + '|' + (p.version || '');
      }).sort().join('||') + '#' + App.dashboard._peerSort + ':' + App.dashboard._peerSortDir;
      if (App.dashboard._lastPeerRenderSig === renderSig) return;
      App.dashboard._lastPeerRenderSig = renderSig;

      // If the set of hosted shards changed since the last snapshot, refresh
      // pipeline plans on visible matrix cards. The initial card render often
      // 404s against /pipeline-plan because peers haven't announced shards
      // yet; without this the path stays invisible until the user toggles
      // list↔matrix views (which re-fetches the plan).
      var prev = App.dashboard._lastPeers || [];
      var shardSig = function(ps) {
        return (ps || []).map(function(p) {
          return (p.node_id || '') + ':' + (p.hosted_shards || 0);
        }).sort().join('|');
      };
      var changed = shardSig(prev) !== shardSig(peers);
      App.dashboard._lastPeers = peers || [];
      if (changed) {
        if (App.dashboard._pipelinePlanRefreshTimer) {
          clearTimeout(App.dashboard._pipelinePlanRefreshTimer);
        }
        App.dashboard._pipelinePlanRefreshTimer = setTimeout(function() {
          document.querySelectorAll('.model-card').forEach(function(card) {
            if (card.offsetParent === null) return;
            App.dashboard._applyPipelinePlan(card);
          });
        }, 500);
      }

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
        '<th data-peer-sort="shards"' + _thClass('shards') + '>' + U.escapeHtml(I18n.t('dashboard.info_shards')) + _sortArrow('shards') + '</th>' +
        '<th data-peer-sort="trust"' + _thClass('trust') + '>' + U.escapeHtml(I18n.t('dashboard.peer_col_trust')) + _sortArrow('trust') + '</th>' +
        '<th data-peer-sort="status"' + _thClass('status') + '>' + U.escapeHtml(I18n.t('dashboard.section_status')) + _sortArrow('status') + '</th>' +
        '</tr></thead><tbody>';

      sorted.forEach(function(p) {
        var name = p.nickname || (p.node_id || 'unknown').substring(0, 12);
        var idSub = p.nickname ? '<span class="peer-id-sub">' + (p.node_id || '').substring(0, 8) + '</span>' : '';
        // Every peer gets exactly one type tag (Pool > LAN > Internet) so it's
        // never ambiguous whether a peer is your own pool device, on your local
        // network, or out on the internet. Matches the backend taxonomy.
        var lanBadge = p.is_pool_member
          ? ' <span class="badge badge-green">' + U.escapeHtml(I18n.t('dashboard.peer_type_pool')) + '</span>'
          : (p.is_lan_peer
              ? ' <span class="badge badge-purple lan-badge">' + U.escapeHtml(I18n.t('dashboard.peer_type_lan')) + '</span>'
              : ' <span class="badge badge-blue">' + U.escapeHtml(I18n.t('dashboard.peer_type_remote')) + '</span>');
        // An anchor holds no shards and serves no inference by design, so
        // without a label it reads as a broken peer. Shown in ADDITION to the
        // location tag rather than replacing it — where the anchor lives still
        // matters for latency.
        if (p.is_anchor) {
          lanBadge += ' <span class="badge badge-amber" title="' +
            U.escapeHtml(I18n.t('dashboard.peer_anchor_tip')) + '">' +
            U.escapeHtml(I18n.t('dashboard.peer_type_anchor')) + '</span>';
        }
        var dotClass = p.healthy ? 'online' : 'degraded';
        var latency = p.latency_ms ? p.latency_ms + 'ms' : '\u2014';
        var shards = p.hosted_shards || 0;
        var trust = p.trust_score !== undefined ? (p.trust_score * 100).toFixed(0) + '%' : '\u2014';
        var status = p.healthy ? I18n.t('dashboard.health_healthy') : I18n.t('dashboard.peer_degraded');
        // Meta line under the peer name: version + GPU. Version is gossiped by
        // every node and makes it obvious at a glance when a peer is on an older
        // build (a real help when a bug behaves differently across machines).
        var verText = p.version ? 'v' + U.escapeHtml(p.version) : '';
        var hwText = '';
        if (p.gpu) {
          // Vendor marker from the reported adapter name. Purely cosmetic, and
          // the full name stays in the text, so an unrecognised vendor just
          // gets the neutral mark rather than a wrong one.
          var g = String(p.gpu).toLowerCase();
          var vendor = g.indexOf('nvidia') >= 0 || g.indexOf('geforce') >= 0 || g.indexOf('rtx') >= 0 || g.indexOf('quadro') >= 0 || g.indexOf('tesla') >= 0
            ? '\u25B2'
            : (g.indexOf('amd') >= 0 || g.indexOf('radeon') >= 0
                ? '\u25CF'
                : (g.indexOf('intel') >= 0 || g.indexOf('arc') >= 0
                    ? '\u25A0'
                    : (g.indexOf('apple') >= 0 ? '\u25C6' : '\u25AB')));
          hwText = '<span title="' + U.escapeHtml(I18n.t('dashboard.gpu_label')) + ': ' + U.escapeHtml(p.gpu) + '">' +
            vendor + ' ' + U.escapeHtml(p.gpu) + '</span>';
        } else if (p.version) {
          // Only claim "CPU" once the peer has actually reported a capability
          // (version is part of it) — otherwise we would label a peer we simply
          // have not heard from yet.
          //
          // Name the processor when the peer sent one, the same way a graphics
          // card is named. "CPU" on its own said nothing — a fanless mini-PC
          // and a sixteen-core server were indistinguishable, and one of them
          // may be the fastest machine on the network. Older peers send no
          // processor details and keep the bare label.
          var cpuText = I18n.t('dashboard.cpu_label');
          var cpuTip = I18n.t('dashboard.peer_cpu_only_tip');
          if (p.cpu && p.cpu.name) {
            cpuText = U.escapeHtml(p.cpu.name);
            if (p.cpu.cores) cpuText += ' ' + U.escapeHtml(I18n.t('hw.cores', { cores: p.cpu.cores }));
            cpuTip += ' \u2014 ' + p.cpu.name;
          }
          hwText = '<span title="' + U.escapeHtml(cpuTip) + '">\u2699 ' + cpuText + '</span>';
        }
        var metaParts = [verText, hwText].filter(Boolean);
        var gpu = metaParts.length
          ? '<div class="text-muted" style="font-size:0.62rem">' + metaParts.join(' · ') + '</div>'
          : '';

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
