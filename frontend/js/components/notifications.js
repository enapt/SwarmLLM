'use strict';

// ============================================================================
// SwarmLLM — Notifications Component
// Toasts, banners, WebSocket, polling, provider health, prune, schedule
// ============================================================================

(function() {
  var S = App.state;
  var U = App.utils;

  // --- Activity + Network Logs (persisted to sessionStorage) ---
  var ACTIVITY_STORAGE_KEY = App.ACTIVITY_KEY;
  var NETWORK_STORAGE_KEY = App.NETWORK_LOG_KEY;
  var _modelsChangedTimer = null;
  var _suppressToasts = true; // suppress toasts during WS history replay
  var _activityEntries = (function() {
    try { var s = sessionStorage.getItem(ACTIVITY_STORAGE_KEY); if (s) return JSON.parse(s); } catch (e) {}
    return [];
  })();
  var _networkEntries = (function() {
    try { var s = sessionStorage.getItem(NETWORK_STORAGE_KEY); if (s) return JSON.parse(s); } catch (e) {}
    return [];
  })();
  var MAX_ACTIVITY = 200;
  var MAX_DISPLAY = 40;

  // Events that go to the Network panel instead of Activity
  var NETWORK_KINDS = {
    'peer_connected': true, 'peer_disconnected': true,
    'shard_announced': true, 'lan_peer_discovered': true,
    'rebalance_peer_left': true,
  };

  // Debounce sessionStorage writes — _pushEntry can fire many times per
  // second during inference. Synchronous setItem + JSON.stringify of up to
  // 100 entries on every event was measurable main-thread cost. The data
  // is only used for page-reload recovery, so a 1-second flush delay is
  // not user-visible.
  var _persistActivityRef = { t: null };
  var _persistNetworkRef = { t: null };
  // Debounce sessionStorage writes — coalesce bursts of activity/network
  // events into one write per second per stream.
  function _debouncedPersist(ref, key, getEntries) {
    if (ref.t) return;
    ref.t = setTimeout(function() {
      ref.t = null;
      try { sessionStorage.setItem(key, JSON.stringify(getEntries().slice(0, 100))); } catch (e) {}
    }, 1000);
  }
  function _persistActivity() {
    _debouncedPersist(_persistActivityRef, ACTIVITY_STORAGE_KEY, function() { return _activityEntries; });
  }
  function _persistNetwork() {
    _debouncedPersist(_persistNetworkRef, NETWORK_STORAGE_KEY, function() { return _networkEntries; });
  }

  // Category → icon mapping for backend activity events
  var ACTIVITY_ICONS = {
    'peer_connected': '\uD83D\uDD17',      // 🔗
    'peer_disconnected': '\u26D4',          // ⛔
    'model_discovered': '\u2728',           // ✨
    'shard_announced': '\uD83D\uDCE1',     // 📡
    'shard_download_started': '\u2B07\uFE0F', // ⬇️
    'shard_download_complete': '\u2705',    // ✅
    'hf_download_complete': '\u2705',       // ✅
    'inference_completed': '\u26A1',        // ⚡
    'inference_failed': '\u274C',           // ❌
    'model_unloaded': '\uD83D\uDCE4',      // 📤
    'shard_loaded_memory': '\uD83D\uDCE5', // 📥
    'shard_unloaded_memory': '\uD83D\uDCE4', // 📤
    'shard_deleted': '\uD83D\uDDD1',       // 🗑
    'shard_pruned': '\u2702\uFE0F',        // ✂️
    'pool_device_joined': '\uD83D\uDD17', // 🔗
    'pool_device_left': '\u26D4',         // ⛔
    'shard_scan_found': '\uD83D\uDD0D',  // 🔍
    'rebalance_peer_left': '\u2696\uFE0F', // ⚖️
    'inference_request': '\u2728',         // ✨
    'daemon_started': '\uD83D\uDE80',     // 🚀
    'shard_verified': '\u2705',           // ✅
    'shard_verification_failed': '\u274C',      // ❌
    'shard_download_failed': '\u274C',    // ❌
    'shard_download_p2p': '\uD83D\uDD17', // 🔗
    'shard_no_source': '\u26A0\uFE0F',   // ⚠️
    'model_download_complete': '\uD83C\uDF89', // 🎉
    'shard_transfer_failed': '\u274C',    // ❌
    'shard_p2p_complete': '\u2705',       // ✅
    'model_loaded': '\uD83E\uDDE0',      // 🧠
    'model_load_skipped': '\u26A0\uFE0F', // ⚠️
    'worker_spawned': '\uD83D\uDCE5',    // 📥
    'worker_unloaded': '\uD83D\uDCE4',   // 📤
    'lan_peer_discovered': '\uD83C\uDF10', // 🌐
    'hf_download_failed': '\u274C',      // ❌
    'shard_write_failed': '\u274C',      // ❌
    'shard_finalize_failed': '\u274C',   // ❌
    'pool_created': '\u2795',            // ➕
    'pool_member_removed': '\u26D4',     // ⛔
    'config_updated': '\u2699\uFE0F',    // ⚙️
    'download_slow': '\u26A0\uFE0F',    // ⚠️
  };

  // Category CSS class for color coding
  var ACTIVITY_CATEGORY_CLASS = {
    'network': 'activity-cat-network',
    'model': 'activity-cat-model',
    'download': 'activity-cat-download',
    'inference': 'activity-cat-inference',
    'auto_manage': 'activity-cat-automanage',
    'pool': 'activity-cat-network',
    'system': 'activity-cat-model',
  };

  function _pushEntry(entry, isNetwork) {
    var list = isNetwork ? _networkEntries : _activityEntries;
    list.unshift(entry);
    if (list.length > MAX_ACTIVITY) list.pop();
    if (isNetwork) { _persistNetwork(); _renderNetworkLog(); }
    else { _persistActivity(); _renderActivityLog(); }
  }

  function logActivity(icon, text, category, modelId) {
    var ts = Date.now();
    var entry = { icon: icon, text: text, ts: ts, category: category || '', modelId: modelId || '' };
    _pushEntry(entry, category === 'network');
  }

  // Format activity event text — try i18n key, fall back to backend English message
  function _formatEventText(data) {
    var key = 'activity.' + data.kind;
    var params = {
      model: data.model_name || data.model_id || '',
      node: data.node_id ? data.node_id.substring(0, 8) : '',
      shard: data.shard_index != null ? (data.shard_index === App.MMPROJ_SHARD_INDEX ? 'mmproj' : String(data.shard_index + 1)) : (data.detail_num != null ? String(data.detail_num) : ''),
      detail: data.detail_str || '',
      count: data.detail_num != null ? String(data.detail_num) : '',
    };
    var translated = I18n.t(key, params);
    // If i18n returned the key itself (no translation), fall back to backend message
    return (translated !== key) ? translated : (data.message || data.kind);
  }

  function _handleActivityEvent(data) {
    var icon = ACTIVITY_ICONS[data.kind] || '\uD83D\uDD35'; // 🔵 default
    var text = _formatEventText(data);
    var category = data.category || '';
    var modelId = data.model_id || '';
    var ts = Date.now();

    // Dedup: skip if the same text appears in the last 10 entries within 30s.
    // Covers WS reconnect replay, rapid-fire identical events, and interleaved
    // duplicates (e.g. cycle events arriving between per-shard events).
    var targetList = NETWORK_KINDS[data.kind] ? _networkEntries : _activityEntries;
    var scan = Math.min(targetList.length, 10);
    for (var i = 0; i < scan; i++) {
      if (targetList[i].text === text && (ts - targetList[i].ts) < 30000) {
        return;
      }
    }

    var entry = { icon: icon, text: text, ts: ts, category: category, modelId: modelId };

    _pushEntry(entry, !!NETWORK_KINDS[data.kind]);

    // --- Unified toast: backend controls when toasts appear via toast_level ---
    // Suppress toasts during WS reconnect history replay (first 2s after connect)
    if (data.toast_level && !_suppressToasts) {
      var toastType = data.toast_level === 'warn' ? 'warning' : (data.toast_level || 'info');
      var toastMs = data.toast_duration_ms || 5000;
      showToast(text, toastType, toastMs);
    }

    // --- Prune history: shard_pruned events carry structured prune data ---
    if (data.kind === 'shard_pruned' && App.pruneSchedule && App.pruneSchedule.prependHistory) {
      App.pruneSchedule.prependHistory({
        model_id: data.model_id,
        model_name: data.model_name,
        shard_index: data.shard_index,
        freed_bytes: data.freed_bytes || data.detail_num || 0,
        holder_count_before: data.holder_count_before,
        holder_count_after: data.holder_count_after,
        remaining_local_shards: data.remaining_local_shards,
        timestamp: data.timestamp,
      });
    }

    // Route to per-model ticker if model_id is present (skipGlobal=true to avoid double-logging)
    if (modelId && App.dashboard && App.dashboard._logModelEvent) {
      App.dashboard._logModelEvent(modelId, icon, data.message || data.kind, true, data.kind);
    }

    // Forward auto-manage events to the header status indicator
    if (App.autoManageStatus && App.autoManageStatus.onEvent) {
      App.autoManageStatus.onEvent(data);
    }

    // Auto-refresh pool tab when pool events arrive
    if (data.kind === 'pool_device_joined' || data.kind === 'pool_device_left') {
      if (App.pool && App.pool.load) App.pool.load();
    }
  }

  // Per-log signature of last render — top-entry ts + length. If neither
  // changed since the last render of this log, skip the full innerHTML
  // rebuild entirely (the visible state is already correct).
  var _renderSig = {};
  function _renderEventLog(entries, logId, countId, emptyText) {
    var log = document.getElementById(logId);
    if (!log) return;
    var countEl = document.getElementById(countId);
    if (countEl) countEl.textContent = I18n.t('activity.count', { count: entries.length });

    var topTs = entries.length > 0 ? entries[0].ts : 0;
    var sig = topTs + ':' + entries.length;
    if (_renderSig[logId] === sig) return;
    _renderSig[logId] = sig;

    var html = '';
    var show = entries.slice(0, MAX_DISPLAY);
    for (var i = 0; i < show.length; i++) {
      var e = show[i];
      var catClass = ACTIVITY_CATEGORY_CLASS[e.category] || '';
      var d = new Date(e.ts);
      var clock = ('0' + d.getHours()).slice(-2) + ':' + ('0' + d.getMinutes()).slice(-2) + ':' + ('0' + d.getSeconds()).slice(-2);
      var ago = U.timeAgo(e.ts);
      var timeHtml = i === 0
        ? '<span class="activity-time" title="' + clock + '">' + ago + '</span>'
        : '<span class="activity-time">' + clock + ' <span class="activity-ago">' + ago + '</span></span>';
      html += '<div class="activity-entry ' + catClass + '"><span class="activity-icon">' + U.escapeHtml(e.icon) + '</span>' +
        '<span class="activity-text">' + U.escapeHtml(e.text) + '</span>' +
        timeHtml + '</div>';
    }
    if (entries.length > MAX_DISPLAY) {
      html += '<div class="activity-overflow text-muted" style="font-size:0.7rem;padding:4px 0;text-align:center">' +
        U.escapeHtml(I18n.t('activity.overflow', { count: entries.length - MAX_DISPLAY })) + '</div>';
    }
    log.innerHTML = html || '<div class="text-muted text-sm" style="padding:8px 0">' + U.escapeHtml(emptyText) + '</div>';
  }

  function _renderActivityLog() { _renderEventLog(_activityEntries, 'activity-log', 'activity-count', I18n.t('activity.none')); }
  function _renderNetworkLog() { _renderEventLog(_networkEntries, 'network-log', 'network-log-count', I18n.t('activity.none_network')); }

  // --- Toast System ---
  function showToast(text, type, duration) {
    type = type || 'info';
    duration = duration || 5000;
    var container = document.getElementById('toast-container');
    if (!container) {
      container = document.createElement('div');
      container.id = 'toast-container';
      container.className = 'toast-container';
      document.body.appendChild(container);
    }
    var icons = { success: '\u2713', error: '\u2717', warning: '\u26A0', info: '\u2139' };
    var tmpl = document.getElementById('tmpl-toast');
    if (!tmpl) return;
    var toast = tmpl.content.cloneNode(true).firstElementChild;
    toast.className = 'toast toast-' + type;
    toast.querySelector('.toast-icon').textContent = icons[type] || icons.info;
    toast.querySelector('.toast-text').textContent = text;
    toast.querySelector('.toast-close').addEventListener('click', function() { toast.remove(); });
    container.appendChild(toast);
    requestAnimationFrame(function() { toast.classList.add('toast-show'); });
    var timer = setTimeout(function() {
      toast.classList.remove('toast-show');
      setTimeout(function() { toast.remove(); }, 300);
    }, duration);
    toast.addEventListener('click', function() { clearTimeout(timer); toast.remove(); });
  }

  // --- Update Banner ---
  function showUpdateBanner(data) {
    if (document.getElementById('update-banner')) return;
    var banner = document.createElement('div');
    banner.id = 'update-banner';
    banner.style.cssText = 'position:fixed;top:0;left:0;right:0;z-index:10000;background:var(--yellow, #eab308);color:var(--bg-primary, #0a0e14);padding:0.6rem 1rem;display:flex;align-items:center;justify-content:center;gap:1rem;font-size:0.85rem;font-weight:500;box-shadow:0 2px 8px rgba(0,0,0,0.3)';
    var text = I18n.t('update.available', { from: data.current_version, to: data.latest_version });
    banner.innerHTML = '<span>' + U.escapeHtml(text) + '</span>';
    if (data.downloaded) {
      var applyBtn = document.createElement('button');
      applyBtn.textContent = I18n.t('update.apply_restart');
      applyBtn.style.cssText = 'background:var(--bg-primary, #0a0e14);color:var(--yellow, #eab308);border:none;border-radius:4px;padding:0.3rem 0.8rem;cursor:pointer;font-size:0.8rem;font-weight:600';
      applyBtn.onclick = async function() {
        applyBtn.disabled = true;
        applyBtn.textContent = I18n.t('update.applying');
        try {
          var resp = await App.authFetch('/api/admin/update/apply', { method: 'POST' });
          if (resp.ok) {
            banner.querySelector('span').textContent = I18n.t('update.applied', { version: data.latest_version });
            applyBtn.style.display = 'none';
          } else {
            applyBtn.textContent = I18n.t('update.failed');
            setTimeout(function() { applyBtn.textContent = I18n.t('actions.retry'); applyBtn.disabled = false; }, 3000);
          }
        } catch (e) {
          applyBtn.textContent = I18n.t('update.error');
          setTimeout(function() { applyBtn.textContent = I18n.t('actions.retry'); applyBtn.disabled = false; }, 3000);
        }
      };
      banner.appendChild(applyBtn);
    } else {
      var dlBtn = document.createElement('button');
      dlBtn.textContent = I18n.t('update.download_apply');
      dlBtn.style.cssText = 'background:var(--bg-primary, #0a0e14);color:var(--yellow, #eab308);border:none;border-radius:4px;padding:0.3rem 0.8rem;cursor:pointer;font-size:0.8rem;font-weight:600';
      dlBtn.onclick = async function() {
        dlBtn.disabled = true;
        dlBtn.textContent = I18n.t('update.checking');
        try {
          var resp = await App.authFetch('/api/admin/update/check', { method: 'POST' });
          if (resp.ok) {
            var result = await resp.json();
            if (result.status === 'update_available' && result.info && result.info.downloaded) {
              dlBtn.textContent = I18n.t('update.applying');
              var applyResp = await App.authFetch('/api/admin/update/apply', { method: 'POST' });
              if (applyResp.ok) {
                banner.querySelector('span').textContent = I18n.t('update.applied', { version: data.latest_version });
                dlBtn.style.display = 'none';
              }
            }
          }
        } catch (e) {
          dlBtn.textContent = I18n.t('update.error');
        }
        setTimeout(function() { dlBtn.textContent = I18n.t('update.download_apply'); dlBtn.disabled = false; }, 3000);
      };
      banner.appendChild(dlBtn);
    }
    document.body.prepend(banner);
  }

  // --- WebSocket ---
  function showWsBanner(type, text) {
    var banner = document.getElementById('ws-banner');
    if (!banner) return;
    if (S.wsBannerTimer) { clearTimeout(S.wsBannerTimer); S.wsBannerTimer = null; }
    banner.innerHTML = '<div class="ws-banner-' + U.escapeHtml(type) + '">' + U.escapeHtml(text) + '</div>';
    banner.classList.add('show');
    setDashboardCover(type !== 'connected');
  }

  function hideWsBanner(delay) {
    var banner = document.getElementById('ws-banner');
    if (!banner) return;
    if (S.wsBannerTimer) clearTimeout(S.wsBannerTimer);
    S.wsBannerTimer = setTimeout(function() {
      banner.classList.remove('show');
      setDashboardCover(false);
    }, delay || 0);
  }

  function setDashboardCover(show) {
    var cover = document.getElementById('dashboard-offline-cover');
    if (!cover) return;
    if (show) {
      cover.classList.add('visible');
      var msg = S.wsWasConnected ? I18n.t('connection.reconnecting') : I18n.t('connection.connecting');
      var sub = S.wsWasConnected ? I18n.t('connection.lost') : I18n.t('connection.waiting');
      cover.querySelector('.cover-msg').textContent = msg;
      cover.querySelector('.cover-sub').textContent = sub;
    } else {
      cover.classList.remove('visible');
    }
  }

  async function connectWebSocket() {
    // Guard against stacking parallel reconnects
    if (S.ws && (S.ws.readyState === WebSocket.CONNECTING || S.ws.readyState === WebSocket.OPEN)) return;
    if (!S.wsWasConnected) setDashboardCover(true);
    // Obtain a short-lived single-use ticket (Bearer-authed POST) — the
    // browser can't set an Authorization header on the WS upgrade, so we
    // pass the ticket as ?t=<hex>. The ticket is atomically consumed and
    // expires in 30 seconds server-side.
    var ticket = '';
    try {
      var tr = await App.authFetch('/api/admin/ws-ticket', { method: 'POST' });
      if (tr.ok) {
        var tj = await tr.json();
        ticket = tj && tj.ticket ? tj.ticket : '';
      }
    } catch (_e) {
      // Fall through — WS will be rejected 401 and the reconnect loop
      // will retry, which surfaces the connection-lost banner via the
      // onclose path. Don't log: this fires on every transient network
      // blip and the user-visible banner is the right surface.
    }
    if (!ticket) {
      // Reconnect path will try again after backoff. Don't hard-error the
      // dashboard — stats_update + activity_event just won't arrive.
      setTimeout(function() { connectWebSocket(); }, 3000);
      return;
    }
    var protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    S.ws = new WebSocket(protocol + '//' + window.location.host + '/api/admin/ws?t=' + encodeURIComponent(ticket));

    S.ws.onopen = function() {
      _suppressToasts = true;
      setTimeout(function() { _suppressToasts = false; }, 2000);
      setDashboardCover(false);
      if (typeof NeuralBg !== 'undefined') NeuralBg.setHealth(1.0);
      S.pollTimers.forEach(function(t) { clearInterval(t); });
      S.pollTimers = [];
      if (S.wsWasConnected) {
        showWsBanner('connected', I18n.t('connection.connected'));
        hideWsBanner(2000);
        // Reload ALL data on reconnect — reset debounce so it's never skipped
        App.dashboard._lastLoadTime = 0;
        App.dashboard.loadInitial();
        App.providerHealth.startHealthPolling();
      } else {
        // First connect this page load — clear stale events from previous session
        _activityEntries = [];
        _networkEntries = [];
        _persistActivity();
        _persistNetwork();
        _renderActivityLog();
        _renderNetworkLog();
      }
      S.wsWasConnected = true;
      logActivity('\u{1F4E1}', I18n.t('activity.connected'), 'system');
    };

    S.ws.onmessage = function(event) {
      try {
        var msg = JSON.parse(event.data);
        if (msg.type === 'stats_update') {
          // Download activity logging is handled by activity_event messages —
          // stats_update only drives UI progress bars and data refreshes.
          App.dashboard.updateStats(msg.data);
          if (msg.data.acquisitions) App.dashboard.updateAcquisitionProgress(msg.data.acquisitions);
          if (App.autoManageStatus) App.autoManageStatus.updateFromStats(msg.data.acquisitions || []);
          App.dashboard.updateShardsLive(msg.data.acquisitions, msg.data.shard_registry || null, msg.data.peer_downloads || null);
          App.downloads.updateFromWs(msg.data.acquisitions);
          U.updateChatDownloadProgress(msg.data.acquisitions);
          if (msg.data.region_summary && S.activeTab === 'network-map') {
            App.networkMap.updateFromWs(msg.data.region_summary);
          }
          // R111: feed wishlist + capacity into the Swarm tab.
          if (App.swarmTab && typeof App.swarmTab.onStats === 'function') {
            App.swarmTab.onStats(msg.data);
          }
        } else if (msg.type === 'update_available') {
          showUpdateBanner(msg.data);
        } else if (msg.type === 'peer_list') {
          App.dashboard.renderPeers((msg.data && msg.data.peers) || []);
        } else if (msg.type === 'activity_event') {
          _handleActivityEvent(msg.data || {});
        } else if (msg.type === 'models_changed') {
          // Refresh model list — activity logging is handled by activity_event messages
          if (_modelsChangedTimer) clearTimeout(_modelsChangedTimer);
          _modelsChangedTimer = setTimeout(function() {
            App.data.invalidateDedup('models');
            App.data.invalidateDedup('providers');
            App.data.cache.cloudModels = [];
            App.models.load();
            App.networkStatus.load();
          }, 1000);
        }
      } catch (e) {}
    };

    S.ws.onclose = function() {
      
      if (typeof NeuralBg !== 'undefined') NeuralBg.setHealth(0.3);
      if (S.wsWasConnected) {
        showWsBanner('disconnected', I18n.t('errors.connection_lost'));
        var peersEl = document.getElementById('stat-peers');
        if (peersEl) peersEl.textContent = '0';
        App.dashboard.renderPeers([]);
      }
      startPolling();
      setTimeout(connectWebSocket, 3000);
    };
    S.ws.onerror = function() { S.ws.close(); };
  }

  function startPolling() {
    if (S.pollTimers.length > 0) return;
    S.pollTimers.push(setInterval(function() { App.dashboard.loadInitial(); }, 30000));
  }

  // --- Provider Health ---
  function getHealthInterval() {
    try { var v = parseInt(localStorage.getItem(App.HEALTH_INTERVAL_KEY)); return v > 0 ? v : 30; } catch(e) { return 30; }
  }

  async function fetchProviderHealth() {
    try {
      var resp = await App.authFetch('/api/admin/provider-health');
      if (!resp.ok) return;
      var data = await resp.json();
      var now = Date.now();
      (data.providers || []).forEach(function(p) {
        S.providerHealth[p.provider] = {
          status: p.status, latency_ms: p.latency_ms,
          detail: p.detail || '', last_checked: now
        };
      });
      App.providerHealth.updateBadges();
      App.networkStatus.load();
    } catch (e) {}
  }

  function startHealthPolling() {
    if (S.healthTimer) clearInterval(S.healthTimer);
    var intervalSec = getHealthInterval();
    if (intervalSec <= 0) return;
    fetchProviderHealth();
    S.healthTimer = setInterval(fetchProviderHealth, intervalSec * 1000);
  }

  // --- Provider Health Badges ---
  // Latency-tier class helpers. Three call sites previously inlined the
  // same 500ms / 2000ms thresholds. Two distinct class systems are in use:
  //   - `dot-ok / dot-slow / dot-down` for the integrated health bar
  //   - `health-fast / health-ok / health-slow` for card + dropdown badges
  // The non-up status mapping diverges across sites and stays inline.
  function _dotLatencyTier(ms) {
    if (ms < 500) return 'dot-ok';
    if (ms < 2000) return 'dot-slow';
    return 'dot-down';
  }
  function _healthLatencyTier(ms) {
    if (ms < 500) return 'health-fast';
    if (ms < 2000) return 'health-ok';
    return 'health-slow';
  }
  // Map a raw backend health status string to a human-readable label
  // for tooltip text. Falls back to the raw status if no translation
  // exists (defensive — new backend states won't render as "undefined").
  function _statusLabel(status) {
    if (status === 'up') return I18n.t('provider.up') || 'OK';
    if (status === 'rate_limited') return I18n.t('provider.rate_limited');
    if (status === 'timeout') return I18n.t('provider.timeout');
    if (status === 'auth_error') return I18n.t('provider.auth_error');
    if (status === 'overloaded') return I18n.t('provider.overloaded');
    if (status === 'down') return I18n.t('provider.down');
    return status;
  }
  App.providerHealth = {
    // Render the integrated provider health bar
    updateHealthBar: function() {
      var bar = document.getElementById('provider-health-bar');
      if (!bar) return;
      var configured = Object.keys(S.providerHealth);

      // Remove old dynamic items (keep #ph-claude-code)
      var oldItems = bar.querySelectorAll('.ph-item.ph-dynamic');
      for (var i = 0; i < oldItems.length; i++) oldItems[i].remove();

      if (configured.length === 0 && bar.querySelector('#ph-claude-code.hidden')) {
        bar.classList.add('hidden');
        return;
      }
      bar.classList.remove('hidden');

      configured.sort().forEach(function(p) {
        var h = S.providerHealth[p];
        var item = document.createElement('div');
        item.className = 'ph-item ph-dynamic';
        var dotClass = 'dot-down';
        var latencyText = '';
        if (h.status === 'up') {
          dotClass = _dotLatencyTier(h.latency_ms);
          latencyText = h.latency_ms + 'ms';
        } else if (h.status === 'rate_limited') { dotClass = 'dot-slow'; latencyText = I18n.t('provider.limited'); }
        else if (h.status === 'timeout') { dotClass = 'dot-down'; latencyText = I18n.t('provider.timeout'); }
        else if (h.status === 'auth_error') { dotClass = 'dot-error'; latencyText = I18n.t('provider.key_invalid'); }
        else { dotClass = 'dot-error'; latencyText = I18n.t('provider.down'); }
        var name = PROVIDER_NAMES[p] || p;
        item.innerHTML =
          '<span class="ph-icon">' + providerIconHtml(p, 14) + '</span>' +
          '<span class="ph-name">' + U.escapeHtml(name) + '</span>' +
          '<span class="ph-dot ' + dotClass + '"></span>' +
          (latencyText ? '<span class="ph-latency">' + U.escapeHtml(latencyText) + '</span>' : '') +
          (typeof isSubscriptionProvider === 'function' && isSubscriptionProvider(p)
            ? '<span class="ph-tag tag-sub">' + I18n.t('dashboard.chip_subscription') + '</span>'
            : '<span class="ph-tag tag-api">' + I18n.t('mode.api') + '</span>');
        item.title = name + ': ' + _statusLabel(h.status) + (h.detail ? ' \u2014 ' + h.detail : '');
        bar.appendChild(item);
      });
    },

    updateBadges: function() {
      App.providerHealth.updateHealthBar();
      function applyCardBadge(badge, h) {
        var statusIcon, statusClass;
        if (h.status === 'up') {
          statusIcon = h.latency_ms + 'ms';
          statusClass = _healthLatencyTier(h.latency_ms);
        } else if (h.status === 'rate_limited') { statusIcon = I18n.t('provider.rate_limited'); statusClass = 'health-warn'; }
        else if (h.status === 'timeout') { statusIcon = I18n.t('provider.timeout'); statusClass = 'health-down'; }
        else if (h.status === 'auth_error') { statusIcon = I18n.t('provider.auth_error'); statusClass = 'health-down'; }
        else if (h.status === 'overloaded') { statusIcon = I18n.t('provider.overloaded'); statusClass = 'health-warn'; }
        else { statusIcon = I18n.t('provider.error'); statusClass = 'health-down'; }
        badge.className = 'provider-health-badge ' + statusClass;
        badge.textContent = statusIcon;
        badge.title = _statusLabel(h.status) + (h.detail ? ': ' + h.detail : '') + ' (' + h.latency_ms + 'ms)';
      }
      Object.keys(S.providerHealth).forEach(function(p) {
        var h = S.providerHealth[p];
        // Card badge
        var badge = document.getElementById('health-badge-' + p);
        if (!badge) {
          var card = document.querySelector('.cloud-model[data-provider="' + U.cssSafeAttr(p) + '"]');
          if (card) {
            var header = card.querySelector('.model-header');
            if (header) {
              badge = document.createElement('span');
              badge.id = 'health-badge-' + p;
              badge.className = 'provider-health-badge';
              header.querySelector('span:last-child').appendChild(badge);
            }
          }
        }
        if (badge) applyCardBadge(badge, h);

        // Dropdown group badge
        var groupEl = document.querySelector('.model-dropdown-group[data-group="' + p + '"]');
        if (!groupEl) return;
        var existingBadge = groupEl.querySelector('.provider-health-badge');
        if (!existingBadge) {
          var ghead = groupEl.querySelector('.model-dropdown-group-header');
          if (!ghead) return;
          existingBadge = document.createElement('span');
          existingBadge.className = 'provider-health-badge';
          ghead.appendChild(existingBadge);
        }
        if (h.status === 'up') {
          existingBadge.className = 'provider-health-badge ' + _healthLatencyTier(h.latency_ms);
          existingBadge.textContent = h.latency_ms + 'ms';
        } else {
          existingBadge.className = 'provider-health-badge health-down';
          existingBadge.textContent = h.status === 'rate_limited' ? I18n.t('provider.limited') : h.status === 'timeout' ? I18n.t('provider.timeout') : I18n.t('provider.down');
        }
      });
    },

    probe: function(modelIds) {
      var now = Date.now();
      var toProbe = modelIds.filter(function(id) {
        if (S._modelStatusPending[id]) return false;
        var cached = S.modelStatus[id];
        if (cached && (now - cached.ts) < 60000) return false;
        return true;
      });
      if (toProbe.length === 0) return;
      toProbe = toProbe.slice(0, 20);
      toProbe.forEach(function(id) { S._modelStatusPending[id] = true; });

      App.authFetch('/api/admin/provider-model-status', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ models: toProbe }),
      }).then(function(resp) {
        if (!resp.ok) return;
        return resp.json();
      }).then(function(data) {
        if (!data || !data.models) return;
        var ts = Date.now();
        data.models.forEach(function(m) {
          S.modelStatus[m.model] = { status: m.status, latency_ms: m.latency_ms, ts: ts };
          delete S._modelStatusPending[m.model];
        });
        try { sessionStorage.setItem(App.MODEL_STATUS_KEY, JSON.stringify(S.modelStatus)); } catch (e) {}
        App.providerHealth.updateModelBadges();
      }).catch(function() {
        toProbe.forEach(function(id) { delete S._modelStatusPending[id]; });
      });
    },

    modelBadgeHtml: function(modelId) {
      var s = S.modelStatus[modelId];
      if (!s) return '';
      function badge(cls, title, text) {
        return '<span class="model-status-badge ' + cls + '" title="' + U.escapeHtml(title) + '">' + U.escapeHtml(text) + '</span>';
      }
      if (s.status === 'up') {
        var cls = s.latency_ms < 1000 ? 'health-fast' : s.latency_ms < 3000 ? 'health-ok' : 'health-slow';
        return badge(cls, I18n.t('provider.responded_in', { ms: s.latency_ms }), String(s.latency_ms) + 'ms');
      }
      if (s.status === 'timeout') return badge('health-slow', I18n.t('provider.model_timeout'), I18n.t('provider.slow'));
      if (s.status === 'unavailable') return badge('health-down', I18n.t('provider.model_unavailable'), I18n.t('provider.down'));
      if (s.status === 'not_found') return badge('health-down', I18n.t('provider.model_not_found'), I18n.t('provider.not_available'));
      if (s.status === 'rate_limited') return badge('health-warn', I18n.t('provider.rate_limited'), I18n.t('provider.limited'));
      return badge('health-down', I18n.t('provider.error'), I18n.t('provider.err'));
    },

    updateModelBadges: function() {
      document.querySelectorAll('.cloud-model-row[data-select-cloud]').forEach(function(row) {
        var modelId = row.getAttribute('data-select-cloud');
        var pingEl = row.querySelector('.cloud-model-row-ping');
        if (pingEl) pingEl.innerHTML = App.providerHealth.modelBadgeHtml(modelId);
      });
      document.querySelectorAll('.model-dropdown-item[data-value]').forEach(function(item) {
        var modelId = item.getAttribute('data-value');
        var existing = item.querySelector('.model-status-badge');
        var html = App.providerHealth.modelBadgeHtml(modelId);
        if (html) {
          if (existing) { existing.outerHTML = html; } else { item.insertAdjacentHTML('beforeend', ' ' + html); }
        }
      });
    },

    startHealthPolling: startHealthPolling,
  };

  // Render restored logs from sessionStorage immediately
  if (_activityEntries.length > 0) setTimeout(_renderActivityLog, 0);
  if (_networkEntries.length > 0) setTimeout(_renderNetworkLog, 0);

  // Refresh "ago" timestamps every 30 seconds. Force a full rerender of
  // activity / network logs (the per-event signature skip would otherwise
  // leave the in-row "ago" text stale until the next activity event).
  setInterval(function() {
    _renderSig = {};
    _renderActivityLog();
    _renderNetworkLog();
    // Also refresh per-model tickers
    document.querySelectorAll('.model-ticker-time').forEach(function(el) {
      var ts = parseInt(el.getAttribute('data-ts'), 10);
      if (ts) el.textContent = U.timeAgo(ts);
    });
  }, 30000);

  App.notifications = {
    showToast: showToast,
    showUpdateBanner: showUpdateBanner,
    connectWebSocket: connectWebSocket,
    startPolling: startPolling,
    logActivity: logActivity,
  };
})();
