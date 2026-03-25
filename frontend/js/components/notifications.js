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
    'rebalance_peer_left': true, 'rebalance_peer_joined': true, 'rebalance_manual': true,
  };

  function _persistActivity() {
    try { sessionStorage.setItem(ACTIVITY_STORAGE_KEY, JSON.stringify(_activityEntries.slice(0, 100))); } catch (e) {}
  }
  function _persistNetwork() {
    try { sessionStorage.setItem(NETWORK_STORAGE_KEY, JSON.stringify(_networkEntries.slice(0, 100))); } catch (e) {}
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
    'rebalance_peer_joined': '\u2696\uFE0F', // ⚖️
    'rebalance_manual': '\u2696\uFE0F',   // ⚖️
    'inference_request': '\u2728',         // ✨
    'daemon_started': '\uD83D\uDE80',     // 🚀
    'shard_verified': '\u2705',           // ✅
    'shard_verify_failed': '\u274C',      // ❌
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

  function logActivity(icon, text, category, modelId) {
    var ts = Date.now();
    var entry = { icon: icon, text: text, ts: ts, category: category || '', modelId: modelId || '' };
    // Route peer/network events to the Network panel
    if (category === 'network') {
      _networkEntries.unshift(entry);
      if (_networkEntries.length > MAX_ACTIVITY) _networkEntries.pop();
      _persistNetwork();
      _renderNetworkLog();
      return;
    }
    _activityEntries.unshift(entry);
    if (_activityEntries.length > MAX_ACTIVITY) _activityEntries.pop();
    _persistActivity();
    _renderActivityLog();
  }

  // Format activity event text — try i18n key, fall back to backend English message
  function _formatEventText(data) {
    var key = 'activity.' + data.kind;
    var params = {
      model: data.model_name || data.model_id || '',
      node: data.node_id ? data.node_id.substring(0, 8) : '',
      shard: data.shard_index != null ? String(data.shard_index + 1) : (data.detail_num != null ? String(data.detail_num) : ''),
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

    // Dedup: skip if the most recent entry has the same text within 3 seconds.
    // This prevents duplicates from WS reconnect history replay and rapid-fire
    // identical events (e.g. check_and_load_model called twice for same shard).
    var targetList = NETWORK_KINDS[data.kind] ? _networkEntries : _activityEntries;
    if (targetList.length > 0 && targetList[0].text === text && (ts - targetList[0].ts) < 3000) {
      return;
    }

    var entry = { icon: icon, text: text, ts: ts, category: category, modelId: modelId };

    if (NETWORK_KINDS[data.kind]) {
      _networkEntries.unshift(entry);
      if (_networkEntries.length > MAX_ACTIVITY) _networkEntries.pop();
      _persistNetwork();
      _renderNetworkLog();
    } else {
      _activityEntries.unshift(entry);
      if (_activityEntries.length > MAX_ACTIVITY) _activityEntries.pop();
      _persistActivity();
      _renderActivityLog();
    }

    // --- Unified toast: backend controls when toasts appear via toast_level ---
    if (data.toast_level) {
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

    // Auto-refresh pool tab when pool events arrive
    if (data.kind === 'pool_device_joined' || data.kind === 'pool_device_left') {
      if (App.pool && App.pool.load) App.pool.load();
    }
  }

  function _renderActivityLog() {
    var log = document.getElementById('activity-log');
    if (!log) return;
    var countEl = document.getElementById('activity-count');
    if (countEl) countEl.textContent = _activityEntries.length + ' events';

    var html = '';
    var show = _activityEntries.slice(0, MAX_DISPLAY);
    for (var i = 0; i < show.length; i++) {
      var e = show[i];
      var catClass = ACTIVITY_CATEGORY_CLASS[e.category] || '';
      var d = new Date(e.ts);
      var clock = ('0' + d.getHours()).slice(-2) + ':' + ('0' + d.getMinutes()).slice(-2) + ':' + ('0' + d.getSeconds()).slice(-2);
      var ago = U.timeAgo(e.ts);
      var timeHtml = i === 0
        ? '<span class="activity-time" title="' + clock + '">' + ago + '</span>'
        : '<span class="activity-time">' + clock + ' <span class="activity-ago">' + ago + '</span></span>';
      html += '<div class="activity-entry ' + catClass + '"><span class="activity-icon">' + e.icon + '</span>' +
        '<span class="activity-text">' + U.escapeHtml(e.text) + '</span>' +
        timeHtml + '</div>';
    }
    if (_activityEntries.length > MAX_DISPLAY) {
      html += '<div class="activity-overflow text-muted" style="font-size:0.7rem;padding:4px 0;text-align:center">' +
        (_activityEntries.length - MAX_DISPLAY) + ' older events</div>';
    }
    log.innerHTML = html || '<div class="text-muted" style="font-size:0.82rem;padding:8px 0">No activity yet.</div>';
  }

  function _renderNetworkLog() {
    var log = document.getElementById('network-log');
    if (!log) return;
    var countEl = document.getElementById('network-log-count');
    if (countEl) countEl.textContent = _networkEntries.length + ' events';

    var html = '';
    var show = _networkEntries.slice(0, MAX_DISPLAY);
    for (var i = 0; i < show.length; i++) {
      var e = show[i];
      var catClass = ACTIVITY_CATEGORY_CLASS[e.category] || '';
      var d = new Date(e.ts);
      var clock = ('0' + d.getHours()).slice(-2) + ':' + ('0' + d.getMinutes()).slice(-2) + ':' + ('0' + d.getSeconds()).slice(-2);
      var ago = U.timeAgo(e.ts);
      var timeHtml = i === 0
        ? '<span class="activity-time" title="' + clock + '">' + ago + '</span>'
        : '<span class="activity-time">' + clock + ' <span class="activity-ago">' + ago + '</span></span>';
      html += '<div class="activity-entry ' + catClass + '"><span class="activity-icon">' + e.icon + '</span>' +
        '<span class="activity-text">' + U.escapeHtml(e.text) + '</span>' +
        timeHtml + '</div>';
    }
    if (_networkEntries.length > MAX_DISPLAY) {
      html += '<div class="activity-overflow text-muted" style="font-size:0.7rem;padding:4px 0;text-align:center">' +
        (_networkEntries.length - MAX_DISPLAY) + ' older events</div>';
    }
    log.innerHTML = html || '<div class="text-muted" style="font-size:0.82rem;padding:8px 0">No network events yet.</div>';
  }

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
    var text = 'Update available: v' + U.escapeHtml(data.current_version) + ' \u2192 v' + U.escapeHtml(data.latest_version);
    banner.innerHTML = '<span>' + text + '</span>';
    if (data.downloaded) {
      var applyBtn = document.createElement('button');
      applyBtn.textContent = 'Apply & Restart';
      applyBtn.style.cssText = 'background:var(--bg-primary, #0a0e14);color:var(--yellow, #eab308);border:none;border-radius:4px;padding:0.3rem 0.8rem;cursor:pointer;font-size:0.8rem;font-weight:600';
      applyBtn.onclick = async function() {
        applyBtn.disabled = true;
        applyBtn.textContent = 'Applying...';
        try {
          var resp = await App.authFetch('/api/admin/update/apply', { method: 'POST' });
          if (resp.ok) {
            banner.querySelector('span').textContent = 'Update applied! Restart SwarmLLM to use v' + U.escapeHtml(data.latest_version);
            applyBtn.style.display = 'none';
          } else {
            applyBtn.textContent = 'Failed';
            setTimeout(function() { applyBtn.textContent = 'Retry'; applyBtn.disabled = false; }, 3000);
          }
        } catch (e) {
          applyBtn.textContent = 'Error';
          setTimeout(function() { applyBtn.textContent = 'Retry'; applyBtn.disabled = false; }, 3000);
        }
      };
      banner.appendChild(applyBtn);
    } else {
      var dlBtn = document.createElement('button');
      dlBtn.textContent = 'Download & Apply';
      dlBtn.style.cssText = 'background:var(--bg-primary, #0a0e14);color:var(--yellow, #eab308);border:none;border-radius:4px;padding:0.3rem 0.8rem;cursor:pointer;font-size:0.8rem;font-weight:600';
      dlBtn.onclick = async function() {
        dlBtn.disabled = true;
        dlBtn.textContent = 'Checking...';
        try {
          var resp = await App.authFetch('/api/admin/update/check', { method: 'POST' });
          if (resp.ok) {
            var result = await resp.json();
            if (result.status === 'update_available' && result.info && result.info.downloaded) {
              dlBtn.textContent = 'Applying...';
              var applyResp = await App.authFetch('/api/admin/update/apply', { method: 'POST' });
              if (applyResp.ok) {
                banner.querySelector('span').textContent = 'Update applied! Restart SwarmLLM to use v' + U.escapeHtml(data.latest_version);
                dlBtn.style.display = 'none';
              }
            }
          }
        } catch (e) {
          dlBtn.textContent = 'Error';
        }
        setTimeout(function() { dlBtn.textContent = 'Download & Apply'; dlBtn.disabled = false; }, 3000);
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
      var msg = S.wsWasConnected ? 'Reconnecting to SwarmLLM\u2026' : 'Connecting to SwarmLLM\u2026';
      var sub = S.wsWasConnected ? 'Lost connection \u2014 retrying every 3 seconds' : 'Waiting for the daemon to respond';
      cover.querySelector('.cover-msg').textContent = msg;
      cover.querySelector('.cover-sub').textContent = sub;
    } else {
      cover.classList.remove('visible');
    }
  }

  function connectWebSocket() {
    if (!S.wsWasConnected) setDashboardCover(true);
    var protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    S.ws = new WebSocket(protocol + '//' + window.location.host + '/api/admin/ws');

    S.ws.onopen = function() {
      S.wsHealthy = true;
      setDashboardCover(false);
      if (typeof NeuralBg !== 'undefined') NeuralBg.setHealth(1.0);
      S.pollTimers.forEach(function(t) { clearInterval(t); });
      S.pollTimers = [];
      if (S.wsWasConnected) {
        showWsBanner('connected', 'Connected');
        hideWsBanner(2000);
        // Reload ALL data on reconnect so nothing is stale
        App.dashboard.loadInitial();
        App.providerHealth.startHealthPolling();
      } else {
        // First connect this page load — clear stale activity from previous daemon session
        _activityEntries = [];
        _networkEntries = [];
        _persistActivity();
        _persistNetwork();
        try { sessionStorage.removeItem(App.MODEL_EVENTS_KEY); sessionStorage.removeItem(App.MODEL_NET_EVENTS_KEY); } catch (e2) {}
      }
      S.wsWasConnected = true;
      logActivity('\u{1F4E1}', 'Connected to SwarmLLM node', 'system');
    };

    S.ws.onmessage = function(event) {
      try {
        var msg = JSON.parse(event.data);
        if (msg.type === 'stats_update') {
          // Download activity logging is handled by activity_event messages —
          // stats_update only drives UI progress bars and data refreshes.
          App.dashboard.updateStats(msg.data);
          if (msg.data.acquisitions) App.dashboard.updateAcquisitionProgress(msg.data.acquisitions);
          App.dashboard.updateShardsLive(msg.data.acquisitions, msg.data.shard_registry || null, msg.data.peer_downloads || null);
          App.downloads.updateFromWs(msg.data.acquisitions);
          U.updateChatDownloadProgress(msg.data.acquisitions);
          if (msg.data.region_summary && S.activeTab === 'network-map') {
            App.networkMap.updateFromWs(msg.data.region_summary);
          }
        } else if (msg.type === 'update_available') {
          showUpdateBanner(msg.data);
        } else if (msg.type === 'peer_list') {
          App.dashboard.renderPeers((msg.data && msg.data.peers) || []);
        } else if (msg.type === 'activity_event') {
          _handleActivityEvent(msg.data || {});
        } else if (msg.type === 'models_changed') {
          // Refresh model list — activity logging is handled by activity_event messages
          if (window._modelsChangedTimer) clearTimeout(window._modelsChangedTimer);
          window._modelsChangedTimer = setTimeout(function() {
            App.models.load();
            App.modeIndicator.load();
          }, 1000);
        }
      } catch (e) {}
    };

    S.ws.onclose = function() {
      S.wsHealthy = false;
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
      App.modeIndicator.load();
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
  App.providerHealth = {
    updateBannerBadges: function() {
      var strip = document.getElementById('provider-badges');
      if (!strip) return;
      var configured = Object.keys(S.providerHealth);
      if (configured.length === 0) { strip.classList.add('hidden'); return; }
      strip.classList.remove('hidden');
      strip.innerHTML = '';
      var badgeTmpl = document.getElementById('tmpl-provider-badge');
      configured.sort().forEach(function(p) {
        var h = S.providerHealth[p];
        var badge = badgeTmpl.content.cloneNode(true).firstElementChild;
        var isError = (h.status === 'auth_error' || h.status === 'down' || h.status === 'error');
        badge.className = 'provider-badge' + (h.status === 'up' ? ' badge-active' : '') + (isError ? ' badge-error' : '');
        var dotClass = 'dot-down';
        var latencyText = '';
        if (h.status === 'up') {
          dotClass = h.latency_ms < 500 ? 'dot-fast' : h.latency_ms < 2000 ? 'dot-ok' : 'dot-slow';
          latencyText = h.latency_ms + 'ms';
        } else if (h.status === 'rate_limited') { dotClass = 'dot-ok'; latencyText = 'Limited'; }
        else if (h.status === 'timeout') { dotClass = 'dot-slow'; latencyText = 'Timeout'; }
        else if (h.status === 'auth_error') { dotClass = 'dot-error'; latencyText = 'Key Invalid'; }
        else if (h.status === 'overloaded') { dotClass = 'dot-ok'; latencyText = 'Busy'; }
        else { dotClass = 'dot-error'; latencyText = 'Down'; }
        var name = PROVIDER_NAMES[p] || p;
        badge.querySelector('.pb-icon').innerHTML = providerIconHtml(p, 18);
        badge.querySelector('.pb-name').textContent = name;
        badge.querySelector('.pb-dot').className = 'pb-dot ' + dotClass;
        var latencyEl = badge.querySelector('.pb-latency');
        if (latencyText) { latencyEl.textContent = latencyText; } else { latencyEl.remove(); }
        badge.title = name + ': ' + h.status + (h.detail ? ' — ' + h.detail : '') + (h.latency_ms ? ' (' + h.latency_ms + 'ms)' : '');
        (function(providerKey, errored) {
          badge.addEventListener('click', function() {
            App.ui.switchTab('dashboard');
            setTimeout(function() {
              var card = document.querySelector('.cloud-model[data-provider="' + U.cssSafeAttr(providerKey) + '"]');
              if (card) {
                card.scrollIntoView({ behavior: 'smooth', block: 'start' });
                card.classList.add('provider-highlight');
                setTimeout(function() { card.classList.remove('provider-highlight'); }, 1500);
              } else if (errored) {
                App.ui.openSettings(true);
              }
            }, 100);
          });
        }(p, isError));
        strip.appendChild(badge);
      });
    },

    updateBadges: function() {
      App.providerHealth.updateBannerBadges();
      Object.keys(S.providerHealth).forEach(function(p) {
        var h = S.providerHealth[p];
        var badge = document.getElementById('health-badge-' + p);
        if (!badge) {
          var card = document.querySelector('.cloud-model[data-provider="' + U.cssSafeAttr(p) + '"]');
          if (!card) return;
          var header = card.querySelector('.model-header');
          if (!header) return;
          badge = document.createElement('span');
          badge.id = 'health-badge-' + p;
          badge.className = 'provider-health-badge';
          header.querySelector('span:last-child').appendChild(badge);
        }
        var statusIcon, statusClass;
        if (h.status === 'up') {
          statusIcon = h.latency_ms + 'ms';
          statusClass = h.latency_ms < 500 ? 'health-fast' : h.latency_ms < 2000 ? 'health-ok' : 'health-slow';
        } else if (h.status === 'rate_limited') { statusIcon = 'Rate limited'; statusClass = 'health-warn'; }
        else if (h.status === 'timeout') { statusIcon = 'Timeout'; statusClass = 'health-down'; }
        else if (h.status === 'auth_error') { statusIcon = 'Auth error'; statusClass = 'health-down'; }
        else if (h.status === 'overloaded') { statusIcon = 'Overloaded'; statusClass = 'health-warn'; }
        else { statusIcon = 'Error'; statusClass = 'health-down'; }
        badge.className = 'provider-health-badge ' + statusClass;
        badge.textContent = statusIcon;
        badge.title = h.status + (h.detail ? ': ' + h.detail : '') + ' (' + h.latency_ms + 'ms)';
      });

      Object.keys(S.providerHealth).forEach(function(p) {
        var h = S.providerHealth[p];
        var groupEl = document.querySelector('.model-dropdown-group[data-group="' + p + '"]');
        if (!groupEl) return;
        var existingBadge = groupEl.querySelector('.provider-health-badge');
        if (!existingBadge) {
          var header = groupEl.querySelector('.model-dropdown-group-header');
          if (!header) return;
          existingBadge = document.createElement('span');
          existingBadge.className = 'provider-health-badge';
          header.appendChild(existingBadge);
        }
        if (h.status === 'up') {
          existingBadge.className = 'provider-health-badge ' + (h.latency_ms < 500 ? 'health-fast' : h.latency_ms < 2000 ? 'health-ok' : 'health-slow');
          existingBadge.textContent = h.latency_ms + 'ms';
        } else {
          existingBadge.className = 'provider-health-badge health-down';
          existingBadge.textContent = h.status === 'rate_limited' ? 'Limited' : h.status === 'timeout' ? 'Slow' : 'Down';
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
      if (s.status === 'up') {
        var cls = s.latency_ms < 1000 ? 'health-fast' : s.latency_ms < 3000 ? 'health-ok' : 'health-slow';
        return '<span class="model-status-badge ' + cls + '" title="Responded in ' + U.escapeHtml(String(s.latency_ms)) + 'ms">' + U.escapeHtml(String(s.latency_ms)) + 'ms</span>';
      }
      if (s.status === 'timeout') return '<span class="model-status-badge health-slow" title="Model timed out (5s)">Slow</span>';
      if (s.status === 'unavailable') return '<span class="model-status-badge health-down" title="Model unavailable (503)">Down</span>';
      if (s.status === 'not_found') return '<span class="model-status-badge health-down" title="Model not found (404)">N/A</span>';
      if (s.status === 'rate_limited') return '<span class="model-status-badge health-warn" title="Rate limited">Limited</span>';
      return '<span class="model-status-badge health-down" title="Error">Err</span>';
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

  // Refresh "ago" timestamps every 30 seconds
  setInterval(function() {
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
