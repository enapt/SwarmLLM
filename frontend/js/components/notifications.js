'use strict';

// ============================================================================
// SwarmLLM — Notifications Component
// Toasts, banners, WebSocket, polling, provider health, prune, schedule
// ============================================================================

(function() {
  var S = App.state;
  var U = App.utils;

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
      }
      S.wsWasConnected = true;
    };

    S.ws.onmessage = function(event) {
      try {
        var msg = JSON.parse(event.data);
        if (msg.type === 'stats_update') {
          App.dashboard.updateStats(msg.data);
          if (msg.data.acquisitions) App.dashboard.updateAcquisitionProgress(msg.data.acquisitions);
          App.dashboard.updateShardsLive(msg.data.acquisitions, msg.data.shard_registry || null, msg.data.peer_downloads || null);
          App.downloads.updateFromWs(msg.data.acquisitions);
          U.updateChatDownloadProgress(msg.data.acquisitions);
          if (msg.data.region_summary && S.activeTab === 'network-map') {
            App.networkMap.updateFromWs(msg.data.region_summary);
          }
        } else if (msg.type === 'lan_peer_discovered') {
          var count = msg.data.peer_count || 1;
          showToast('Found ' + count + ' peer' + (count !== 1 ? 's' : '') + ' on your local network \u2014 zero configuration needed!', 'success', 8000);
        } else if (msg.type === 'update_available') {
          showUpdateBanner(msg.data);
        } else if (msg.type === 'peer_list') {
          App.dashboard.renderPeers(msg.data.peers || []);
        } else if (msg.type === 'prune_event') {
          var d = msg.data;
          var freed = U.formatBytes(d.freed_bytes || 0);
          var text = 'Pruned shard ' + U.escapeHtml(String(d.shard_index)) + ' of ' + U.escapeHtml(d.model_name || d.model_id) +
            ' \u2014 ' + U.escapeHtml(String(d.holder_count_before)) + '\u2192' + U.escapeHtml(String(d.holder_count_after)) + ' holders (freed ' + U.escapeHtml(freed) + ')';
          showToast(text, 'info', 6000);
          App.pruneSchedule.prependHistory(d);
        } else if (msg.type === 'system_notification') {
          var n = msg.data;
          var level = n.level === 'error' ? 'error' : (n.level === 'warn' ? 'warning' : 'info');
          showToast(n.title + ': ' + n.message, level, 10000);
        } else if (msg.type === 'models_changed') {
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
              var card = document.querySelector('.cloud-model[data-provider="' + providerKey + '"]');
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
          var card = document.querySelector('.cloud-model[data-provider="' + p + '"]');
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
        try { sessionStorage.setItem('swarmllm_model_status', JSON.stringify(S.modelStatus)); } catch (e) {}
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
        return '<span class="model-status-badge ' + cls + '" title="Responded in ' + s.latency_ms + 'ms">' + s.latency_ms + 'ms</span>';
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

  App.notifications = {
    showToast: showToast,
    showUpdateBanner: showUpdateBanner,
    connectWebSocket: connectWebSocket,
    startPolling: startPolling,
  };
})();
