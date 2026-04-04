'use strict';

// ============================================================================
// SwarmLLM — Dashboard Component
// Stats, model cards, peer list, shard grid, acquisition progress
// ============================================================================

(function() {
  var S = App.state;
  var U = App.utils;

  // Per-model event logs — split into activity and network
  var MODEL_EVENTS_KEY = App.MODEL_EVENTS_KEY;
  var MODEL_NET_EVENTS_KEY = App.MODEL_NET_EVENTS_KEY;
  var _modelEvents = (function() {
    try { var s = sessionStorage.getItem(MODEL_EVENTS_KEY); if (s) return JSON.parse(s); } catch (e) {}
    return {};
  })();
  var _modelNetEvents = (function() {
    try { var s = sessionStorage.getItem(MODEL_NET_EVENTS_KEY); if (s) return JSON.parse(s); } catch (e) {}
    return {};
  })();

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
      '<div class="flex-between" style="font-size:0.75rem;margin-bottom:3px">' +
      '<span class="text-muted">' + opts.label + '</span>' +
      right +
      '</div>' +
      '<div class="dl-bar">' + bar + '</div>' +
      '</div>';
  }

  function _persistModelEvents() {
    try {
      var slim = {};
      Object.keys(_modelEvents).slice(0, 20).forEach(function(k) { slim[k] = _modelEvents[k].slice(0, 10); });
      sessionStorage.setItem(MODEL_EVENTS_KEY, JSON.stringify(slim));
    } catch (e) {}
    try {
      var slim2 = {};
      Object.keys(_modelNetEvents).slice(0, 20).forEach(function(k) { slim2[k] = _modelNetEvents[k].slice(0, 10); });
      sessionStorage.setItem(MODEL_NET_EVENTS_KEY, JSON.stringify(slim2));
    } catch (e) {}
  }

  App.dashboard = {
    _peersExpanded: false,
    _lastPeers: [],

    _logModelEvent: function(modelId, icon, text, skipGlobal, kind) {
      var isNet = kind && MODEL_NET_KINDS[kind];
      var store = isNet ? _modelNetEvents : _modelEvents;
      if (!store[modelId]) store[modelId] = [];
      var events = store[modelId];
      var ts = Date.now();
      events.unshift({ icon: icon, text: text, ts: ts });
      if (events.length > 15) events.pop();

      _persistModelEvents();
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
        if (events.length === 0) return '<div class="text-muted" style="font-size:0.68rem;padding:2px 0">' + emptyText + '</div>';
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
      if (data.node_id) document.getElementById('node-id').textContent = data.node_id;
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
      if (data.peers !== undefined) {
        document.getElementById('stat-peers').textContent = data.peers;
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
        document.getElementById('stat-credits').textContent = bal.toLocaleString();
        document.getElementById('credit-balance').textContent = bal.toLocaleString();
        document.getElementById('credit-earned').textContent = '+' + earned.toLocaleString();
        document.getElementById('credit-spent').textContent = '-' + spent.toLocaleString();
        var prevBal = S.creditHistory.length > 0 ? S.creditHistory[S.creditHistory.length - 1]._bal : bal;
        var delta = bal - prevBal;
        S.creditHistory.push({ _bal: bal, v: delta });
        if (S.creditHistory.length > 30) S.creditHistory.shift();
        U.renderSparkline('credit-sparkline', S.creditHistory.map(function(e) { return e.v; }));
      }
      if (data.requests_served !== undefined) document.getElementById('stat-served').textContent = data.requests_served;
      if (data.requests_made !== undefined) document.getElementById('stat-requests-made').textContent = data.requests_made;
      if (data.forwards_served !== undefined) document.getElementById('stat-forwards').textContent = data.forwards_served;
      if (data.active_requests !== undefined) document.getElementById('stat-active').textContent = data.active_requests;

      App.modeIndicator.update(data, S._cachedProviderData);

      if (typeof NeuralBg !== 'undefined') NeuralBg.updateState(data);
    },

    renderModels: function(models, cloudModels) {
      // models cached in App.data.cache.models
      var list = document.getElementById('models-list');
      var empty = document.getElementById('models-empty');
      var loading = document.getElementById('models-loading');
      if (loading) loading.remove();

      var hasCloud = cloudModels && cloudModels.length > 0;
      if ((!models || models.length === 0) && !hasCloud) {
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

      if (models.length === 0 && !hasCloud) {
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
        var statCloudTotal = hasCloud ? cloudModels.length : 0;
        var statProviders = 0;
        if (hasCloud) {
          var _pset = {};
          cloudModels.forEach(function(cm) { _pset[cm.provider || 'cloud'] = 1; });
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
      }

      // Sort swarm models
      var swarmSort = S._swarmModelSort || 'az';
      function _sortModels(arr, mode) {
        var sorted = arr.slice();
        if (mode === 'az') {
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
        card.className = 'model-card' + (isReady ? ' ready' : (isDownloading ? ' downloading' : (isPartial ? ' partial' : '')));
        card.setAttribute('data-model-id', m.id);

        // Status pill — Ready takes priority over Downloading when model is usable
        var statusHtml = '';
        if (m.status === 'loaded') {
          statusHtml = '<span class="model-status-pill active">\u25CF ' + U.escapeHtml(I18n.t('dashboard.status_active')) + '</span>';
        } else if (isReady && !isDownloading) {
          statusHtml = '<span class="model-status-pill ready">' + U.escapeHtml(I18n.t('dashboard.status_ready')) + '</span>';
        } else if (isCachingLocally) {
          statusHtml = '<span class="model-status-pill ready">' + U.escapeHtml(I18n.t('dashboard.status_ready')) + '</span>';
        } else if (isDownloading) {
          statusHtml = '<span class="model-status-pill downloading"><span class="spinner" style="width:10px;height:10px;border-width:1.5px;vertical-align:middle;margin-right:3px"></span>' + U.escapeHtml(I18n.t('dashboard.status_downloading')) + '</span>';
        } else if (isPartial) {
          statusHtml = '<span class="model-status-pill partial">' + U.escapeHtml(I18n.t('dashboard.local_status', { hosted: hostedShards, total: shardCount })) + '</span>';
        } else {
          statusHtml = '<span class="model-status-pill network">' + U.escapeHtml(I18n.t('dashboard.status_on_network')) + '</span>';
        }

        // Trust level badge
        var trustBadge = '';
        if (m.trust_level === 'network_popular') {
          trustBadge = '<span class="badge-trust badge-trust-popular" title="' + U.escapeHtml(I18n.t('dashboard.trust_popular')) + '">' + U.escapeHtml(I18n.t('dashboard.badge_popular')) + '</span>';
        } else if (m.trust_level === 'demand_verified') {
          trustBadge = '<span class="badge-trust badge-trust-verified" title="' + U.escapeHtml(I18n.t('dashboard.trust_verified')) + '">' + U.escapeHtml(I18n.t('dashboard.badge_verified')) + '</span>';
        } else if (m.trust_level === 'pinned') {
          trustBadge = '<span class="badge-trust badge-trust-pinned" title="' + U.escapeHtml(I18n.t('dashboard.trust_pinned')) + '">' + U.escapeHtml(I18n.t('dashboard.badge_pinned')) + '</span>';
        }

        // Encrypted pipeline badge
        var encBadge = '';
        if (m.shard_count > 1 && m.local) {
          var encReady = m.has_first_shard && m.has_last_shard;
          var encActive = m.encrypted_pipeline;
          var encClass = encActive ? 'badge-encrypted active' : (encReady ? 'badge-encrypted ready' : 'badge-encrypted faded');
          var encTitle = encActive ? I18n.t('dashboard.enc_active') :
            (encReady ? I18n.t('dashboard.enc_available') :
              I18n.t('dashboard.enc_unavailable'));
          var missingParts = [];
          if (!m.has_first_shard) missingParts.push(I18n.t('dashboard.enc_missing_first'));
          if (!m.has_last_shard) missingParts.push(I18n.t('dashboard.enc_missing_last', { n: m.shard_count - 1 }));
          if (missingParts.length > 0) encTitle += '. ' + I18n.t('dashboard.enc_missing', { parts: missingParts.join(', ') });
          encBadge = '<span class="' + encClass + '" data-enc-toggle="' + U.escapeHtml(m.id) + '" data-enc-ready="' + (encReady ? '1' : '0') + '" title="' + U.escapeHtml(encTitle) + '">&#128274;</span>';
        }

        // Source label
        var sourceLabel = '';
        if (m.source === 'network' && hostedShards === 0) {
          sourceLabel = '<span class="badge badge-remote" title="' + U.escapeHtml(I18n.t('dashboard.badge_remote')) + '">' + U.escapeHtml(I18n.t('dashboard.badge_remote_label')) + '</span>';
        }

        // Gear + info buttons
        var gearHtml = '<button class="model-gear-btn" data-am-gear="' + U.escapeHtml(m.id) + '" title="' + U.escapeHtml(I18n.t('dashboard.gear_title')) + '">&#9881;</button>';
        var metaBtnHtml = m.has_header ? '<button class="model-meta-btn" data-meta-toggle="' + U.escapeHtml(m.id) + '" title="' + U.escapeHtml(I18n.t('models.metadata_header')) + '">&#9432;</button>' : '';

        // Shard grid
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
          if (hasVram) legendParts.push('<span class="sleg"><span class="sleg-swatch sleg-vram"></span>' + U.escapeHtml(I18n.t('dashboard.active_in', { mem: S._gpuInference ? 'VRAM' : 'RAM' })) + '</span>');
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
          if (networkMissing > 0) { healthLabel = I18n.t('dashboard.health_at_risk'); healthClass = 'health-low'; }
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

        // Footer meta info
        var footerMeta = [];
        footerMeta.push(U.formatBytes(m.total_size_bytes || 0));
        if (shardCount > 0) footerMeta.push(I18n.t(shardCount === 1 ? 'dashboard.shard_count_one' : 'dashboard.shard_count_other', { count: shardCount }));
        if (m.estimated_vram_mb) {
          if (S._gpuInference) {
            footerMeta.push(I18n.t('dashboard.vram_estimate', { vram: U.formatMB(m.estimated_vram_mb) }));
          } else {
            footerMeta.push('<span title="' + U.escapeHtml(I18n.t('hw.low_ram_tip')) + '" style="cursor:help">' + U.escapeHtml(I18n.t('hw.low_ram')) + '</span>');
          }
        }
        if (m.peers_hosting > 0) footerMeta.push(I18n.t('dashboard.peer_count', { count: m.peers_hosting }));
        else if (hostedShards > 0) footerMeta.push('<span style="color:var(--orange)">' + U.escapeHtml(I18n.t('dashboard.local_only')) + '</span>');

        // Missing files warning
        var fileIndicators = '';
        if (hostedShards > 0 || isDownloading) {
          var hasManifest = m.has_manifest !== false;
          var hasHeader = m.has_header !== false;
          if (!hasManifest || !hasHeader) {
            var missingFiles = [];
            if (!hasManifest) missingFiles.push('manifest');
            if (!hasHeader) missingFiles.push('header');
            fileIndicators = '<span style="color:var(--orange);font-size:0.7rem" title="' + U.escapeHtml(I18n.t('dashboard.missing_files', { files: missingFiles.join(', ') })) + '">' + I18n.t('dashboard.missing_warning', { files: missingFiles.join(' + ') }) + '</span>';
          }
        }

        // Action buttons
        var actionHtml = '';
        if (m.status === 'loaded') {
          actionHtml = '<button class="btn btn-sm btn-outline" data-unload-model="' + U.escapeHtml(m.id) + '" title="' + U.escapeHtml(I18n.t('dashboard.unload_tip')) + '">' + U.escapeHtml(I18n.t('dashboard.btn_unload_all')) + '</button>';
        } else if (isReady) {
          actionHtml = '<button class="btn btn-sm btn-primary" data-select-model="' + U.escapeHtml(m.id) + '">' + U.escapeHtml(I18n.t('dashboard.btn_use')) + '</button>';
        } else if (isDownloading) {
          actionHtml = '<button class="shard-cancel-btn" data-cancel-download="' + U.escapeHtml(m.id) + '" title="' + U.escapeHtml(I18n.t('shard.cancel_download')) + '">&times; ' + U.escapeHtml(I18n.t('actions.cancel')) + '</button>';
        } else if (m.source === 'network' || m.status === 'available' || m.status === 'partial') {
          actionHtml = '<button class="btn btn-sm" data-request-model="' + U.escapeHtml(m.id) + '">' + U.escapeHtml(I18n.t('models.download')) + '</button>';
        }

        var removeHtml = '';
        if (hostedShards > 0 && !isDownloading) {
          removeHtml = '<button class="model-remove-btn" data-remove-model="' + U.escapeHtml(m.id) + '">' + U.escapeHtml(I18n.t('actions.remove')) + '</button>';
        }

        var name = U.formatModelDisplayName(m.name || m.id);
        var creatorIconHtml = providerIconHtml(modelIconKey(m.id), 14);

        // Card HTML
        card.innerHTML =
          '<div class="model-card-title">' +
            '<div class="model-card-name-row">' +
              creatorIconHtml +
              '<span class="model-name" title="' + U.escapeHtml(m.id) + '">' + U.escapeHtml(name) + '</span>' +
              encBadge + sourceLabel + trustBadge + healthBadgeHtml +
            '</div>' +
            '<div class="model-card-controls">' +
              statusHtml + metaBtnHtml + gearHtml +
            '</div>' +
          '</div>' +
          '<div class="model-card-shards">' +
            healthBarHtml + shardHtml + progressHtml + perShardDlHtml +
          '</div>' +
          '<div class="model-ticker" data-model-ticker="' + safeId + '" style="display:none"></div>' +
          '<div class="model-card-footer">' +
            '<div class="model-card-meta">' +
              footerMeta.map(function(p) { return '<span>' + p + '</span>'; }).join('') +
              (fileIndicators ? '<span>' + fileIndicators + '</span>' : '') +
            '</div>' +
            '<div class="model-card-actions">' + actionHtml + removeHtml + '</div>' +
          '</div>' +
          '<div class="gguf-metadata-panel hidden" data-meta-panel="' + U.escapeHtml(m.id) + '"></div>';

        if (swarmBody) swarmBody.appendChild(card);

        // Restore per-model activity ticker from stored events (DOM only, don't re-log)
        if (_modelEvents[m.id] && _modelEvents[m.id].length > 0) {
          App.dashboard._renderModelTicker(m.id);
        }
      });

      // Cloud provider models
      if (hasCloud) {
        var byProvider = {};
        cloudModels.forEach(function(cm) {
          var p = cm.provider || 'cloud';
          if (!byProvider[p]) byProvider[p] = [];
          byProvider[p].push(cm);
        });

        function getCtxLen(cm) {
          if (!cm.meta) return 0;
          return cm.meta.context_length || cm.meta.context_window || cm.meta.max_model_len || 0;
        }

        var _nonChatPattern = /dall-e|tts|whisper|embed|moderation|davinci-\d|babbage-\d|text-embedding|audio/i;

        function sortCloudModels(models, sortBy) {
          var sorted = models.slice();
          if (sortBy === 'ctx-desc') {
            sorted.sort(function(a, b) { return getCtxLen(b) - getCtxLen(a); });
          } else if (sortBy === 'ctx-asc') {
            sorted.sort(function(a, b) { return getCtxLen(a) - getCtxLen(b); });
          } else if (sortBy === 'avail') {
            sorted.sort(function(a, b) {
              var sa = S.modelStatus[a.id], sb = S.modelStatus[b.id];
              var rank = { up: 0, rate_limited: 1, timeout: 3, unavailable: 4, not_found: 5, error: 4 };
              var ra = sa ? (rank[sa.status] !== undefined ? rank[sa.status] : 2) : 2;
              var rb = sb ? (rank[sb.status] !== undefined ? rank[sb.status] : 2) : 2;
              if (ra !== rb) return ra - rb;
              var la = sa ? sa.latency_ms : 99999, lb = sb ? sb.latency_ms : 99999;
              return la - lb;
            });
          } else if (sortBy === 'popular') {
            sorted.sort(function(a, b) {
              var aNon = _nonChatPattern.test(a.id) ? 1 : 0;
              var bNon = _nonChatPattern.test(b.id) ? 1 : 0;
              if (aNon !== bNon) return aNon - bNon;
              var ca = (a.meta && a.meta.created) || 0;
              var cb = (b.meta && b.meta.created) || 0;
              if (ca !== cb) return cb - ca;
              var na = (a.name || a.id).toLowerCase(), nb = (b.name || b.id).toLowerCase();
              return na < nb ? -1 : na > nb ? 1 : 0;
            });
          } else {
            sorted.sort(function(a, b) {
              var na = (a.name || a.id).toLowerCase(), nb = (b.name || b.id).toLowerCase();
              return na < nb ? -1 : na > nb ? 1 : 0;
            });
          }
          return sorted;
        }

        function renderCloudRow(cm) {
          var ctx = getCtxLen(cm);
          var ctxStr = ctx > 0 ? (ctx >= 1000 ? Math.round(ctx / 1000) + 'K' : ctx.toString()) : '';
          var pingHtml = App.providerHealth.modelBadgeHtml(cm.id);
          return '<div class="cloud-model-row" data-select-cloud="' + U.escapeHtml(cm.id) + '" title="' + U.escapeHtml(cm.id) + '">' +
            '<span class="cloud-model-row-name">' + U.escapeHtml(cm.name || cm.id) + '</span>' +
            (ctxStr ? '<span class="cloud-model-row-ctx">' + ctxStr + '</span>' : '<span class="cloud-model-row-ctx"></span>') +
            '<span class="cloud-model-row-ping">' + pingHtml + '</span>' +
            '</div>';
        }

        function renderRowsInto(container, models) {
          container.innerHTML = models.length > 0
            ? models.map(renderCloudRow).join('')
            : '<div class="cloud-model-empty">' + U.escapeHtml(I18n.t('dashboard.cloud_no_match')) + '</div>';
        }

        var providerCount = Object.keys(byProvider).length;
        var cloudSection = document.createElement('details');
        cloudSection.className = 'models-section';
        cloudSection.open = true;
        var cloudMeta = I18n.t('dashboard.providers_count', { count: providerCount, models: cloudModels.length });
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
          var pLabel = PROVIDER_NAMES[p] || p;
          var pModels = byProvider[p];
          var sorted = sortCloudModels(pModels, 'popular');
          var filterId = 'cloud-filter-' + p;
          var sortId = 'cloud-sort-' + p;
          var listId = 'cloud-list-wrap-' + p;

          var card = document.createElement('div');
          card.className = 'model-card cloud-model';
          card.setAttribute('data-provider', p);

          var cardIconHtml = providerIconHtml(p, 18);
          card.innerHTML =
            '<div class="cloud-card-header">' +
              '<span class="cloud-provider-name">' + (cardIconHtml ? cardIconHtml + ' ' : '') + U.escapeHtml(pLabel) + '</span>' +
              '<span>' +
                '<span class="badge badge-cloud">' + I18n.t('dashboard.cloud_models_count', { count: pModels.length }) + '</span>' +
                '<span class="cloud-status-ok">\u25cf ' + U.escapeHtml(I18n.t('dashboard.cloud_connected')) + '</span>' +
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
            '<div class="cloud-card-note">' + U.escapeHtml(I18n.t('dashboard.cloud_note', { provider: pLabel })) + '</div>';

          cloudBody.appendChild(card);

          var listContainer = document.getElementById(listId);
          if (listContainer) renderRowsInto(listContainer, sorted);

          var visibleIds = sorted.slice(0, 20).map(function(cm) { return cm.id; });
          setTimeout(function() { App.providerHealth.probe(visibleIds); }, 500);

          var filterEl = document.getElementById(filterId);
          var sortEl = document.getElementById(sortId);
          var refreshRows = function() {
            var query = filterEl ? filterEl.value.toLowerCase().trim() : '';
            var sortBy = sortEl ? sortEl.value : 'popular';
            var filtered = query ? pModels.filter(function(cm) {
              var text = ((cm.name || '') + ' ' + cm.id + ' ' + (cm.meta && cm.meta.owned_by ? cm.meta.owned_by : '')).toLowerCase();
              return text.indexOf(query) !== -1;
            }) : pModels;
            var s = sortCloudModels(filtered, sortBy);
            if (listContainer) renderRowsInto(listContainer, s);
            App.providerHealth.probe(s.slice(0, 20).map(function(cm) { return cm.id; }));
          };
          if (filterEl) {
            filterEl.addEventListener('input', refreshRows);
            filterEl.addEventListener('paste', function() { setTimeout(refreshRows, 0); });
          }
          if (sortEl) sortEl.addEventListener('change', function() {
            refreshRows();
            if (sortEl.value === 'avail') App.providerHealth.probe(pModels.map(function(cm) { return cm.id; }).slice(0, 40));
          });
        });
        if (Object.keys(S.modelStatus).length > 0) App.providerHealth.updateModelBadges();
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
            App.dashboard._logModelEvent(pd.model_id, '\u{1F4E1}', I18n.t('dashboard.peer_downloading_log', { peer: pd.node_id.substring(0, 8), shard: pd.shard_index + 1 }), true);
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

    renderPeers: function(peers) {
      var PEER_LIMIT = 5;
      var list = document.getElementById('peers-list');
      var summary = document.getElementById('peers-summary');
      var overflow = document.getElementById('peers-overflow');
      var pLoading = document.getElementById('peers-loading');
      if (pLoading) pLoading.remove();
      if (!list) return;

      App.dashboard._lastPeers = peers || [];

      if (peers && peers.length > 0) {
        var lanCount = peers.filter(function(p) { return p.is_lan_peer; }).length;
        var healthyCount = peers.filter(function(p) { return p.healthy; }).length;
        if (summary) {
          summary.textContent = I18n.t('dashboard.peers_summary', { count: peers.length, lan: lanCount, healthy: healthyCount });
        }

        list.innerHTML = '';
        var showAll = App.dashboard._peersExpanded;
        var visible = showAll ? peers : peers.slice(0, PEER_LIMIT);
        visible.forEach(function(p) {
          list.appendChild(App.dashboard.renderPeerItem(p));
        });

        if (overflow) {
          if (peers.length > PEER_LIMIT) {
            overflow.style.display = '';
            var btn = document.getElementById('btn-show-all-peers');
            if (btn) btn.textContent = showAll ? I18n.t('dashboard.show_fewer') : I18n.t('dashboard.show_all', { count: peers.length });
          } else {
            overflow.style.display = 'none';
          }
        }
      } else {
        if (summary) summary.textContent = '';
        list.innerHTML = '<div class="text-muted" style="font-size:0.85rem">' + I18n.t('network.no_peers_yet') + '</div>';
        if (overflow) overflow.style.display = 'none';
      }
    },

    loadNetworkData: async function() {
      try {
        var peers = await App.data.loadPeers();
        App.dashboard.renderPeers(peers);
      } catch (e) {
        var list = document.getElementById('peers-list');
        var pLoading2 = document.getElementById('peers-loading');
        if (pLoading2) pLoading2.remove();
        if (list) list.innerHTML = '<div class="text-muted" style="font-size:0.85rem">' + I18n.t('network.no_peers_yet') + '</div>';
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
