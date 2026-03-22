'use strict';

// ============================================================================
// SwarmLLM — UI Module
// Tab switching, sidebar, modals, mode indicator
// ============================================================================

(function() {
  var S = App.state;

  App.ui = {
    switchTab: function(tab, skipHistory) {
      S.activeTab = tab;
      if (!skipHistory) {
        var path = tab === 'chat' ? '/chat'
          : tab === 'leaderboard' ? '/admin/leaderboard'
          : tab === 'network-map' ? '/admin/network'
          : tab === 'compare' ? '/admin/compare'
          : tab === 'devices' ? '/admin/devices'
          : '/admin';
        if (window.location.pathname !== path) {
          history.pushState({ tab: tab }, '', path);
        }
      }
      document.querySelectorAll('.tab-btn').forEach(function(b) {
        b.classList.toggle('active', b.dataset.tab === tab);
      });
      document.getElementById('view-chat').style.display = tab === 'chat' ? '' : 'none';
      document.getElementById('view-dashboard').style.display = tab === 'dashboard' ? '' : 'none';
      var lbView = document.getElementById('view-leaderboard');
      if (lbView) lbView.style.display = tab === 'leaderboard' ? '' : 'none';
      var mapView = document.getElementById('view-network-map');
      if (mapView) mapView.style.display = tab === 'network-map' ? '' : 'none';
      var compareView = document.getElementById('view-compare');
      if (compareView) compareView.style.display = tab === 'compare' ? '' : 'none';
      var devicesView = document.getElementById('view-devices');
      if (devicesView) devicesView.style.display = tab === 'devices' ? '' : 'none';
      var sidebar = document.getElementById('sidebar');
      var edgeTrigger = document.getElementById('sidebar-edge-trigger');
      if (sidebar) {
        if (tab === 'chat') {
          sidebar.style.display = '';
          sidebar.classList.remove('sidebar-float');
          if (window.innerWidth >= 768) sidebar.classList.remove('collapsed');
          if (edgeTrigger) edgeTrigger.classList.remove('active');
        } else {
          sidebar.style.display = '';
          sidebar.classList.add('sidebar-float');
          sidebar.classList.add('collapsed');
          if (edgeTrigger) edgeTrigger.classList.add('active');
        }
      }
      if (tab === 'chat') {
        App.chat.scrollToBottom();
        document.getElementById('chat-input').focus();
      }
      if (tab === 'leaderboard') {
        App.identity.loadLeaderboard();
      }
      if (tab === 'network-map') {
        App.networkMap.refresh();
      }
      if (tab === 'compare' && App.compare) {
        App.compare.loadModels();
        App.compare.renderHistory();
      }
      if (tab === 'devices' && App.pool) {
        App.pool.load();
      }
    },

    openSidebar: function() {
      var sidebar = document.getElementById('sidebar');
      var overlay = document.getElementById('sidebar-overlay');
      if (sidebar) sidebar.classList.remove('collapsed');
      if (overlay && window.innerWidth < 768) overlay.style.display = 'block';
      var btn = document.getElementById('hamburger-btn');
      if (btn) btn.setAttribute('aria-expanded', 'true');
    },

    closeSidebar: function() {
      var sidebar = document.getElementById('sidebar');
      var overlay = document.getElementById('sidebar-overlay');
      if (sidebar) sidebar.classList.add('collapsed');
      if (overlay) overlay.style.display = 'none';
      var btn = document.getElementById('hamburger-btn');
      if (btn) btn.setAttribute('aria-expanded', 'false');
    },

    toggleSidebar: function() {
      var sidebar = document.getElementById('sidebar');
      if (sidebar && !sidebar.classList.contains('collapsed')) {
        App.ui.closeSidebar();
      } else {
        App.ui.openSidebar();
      }
    },

    openSettings: function(scrollToProviders) {
      document.getElementById('settings-modal').classList.remove('hidden');
      App.settings.load();
      if (scrollToProviders) {
        var section = document.getElementById('settings-providers-section');
        if (section) {
          section.open = true;
          setTimeout(function() { section.scrollIntoView({ behavior: 'smooth', block: 'center' }); }, 100);
        }
      }
    },

    closeSettings: function() {
      document.getElementById('settings-modal').classList.add('hidden');
    },

    openModelBrowser: function() {
      document.getElementById('model-browser-modal').classList.remove('hidden');
      var input = document.getElementById('hf-search-input');
      if (input) setTimeout(function() { input.focus(); }, 100);
    },

    closeModelBrowser: function() {
      document.getElementById('model-browser-modal').classList.add('hidden');
    },

    showBanner: function(type, message) {
      App.notifications.showToast(message, type === 'warning' ? 'warning' : type === 'error' ? 'error' : type === 'success' ? 'success' : 'info');
    }
  };

  // --- Mode Indicator ---
  App.modeIndicator = {
    update: function(statsData, providerData) {
      var indicator = document.getElementById('mode-indicator');
      var dot = document.getElementById('mode-dot');
      var label = document.getElementById('mode-label');
      var detail = document.getElementById('mode-detail');
      if (!dot || !label || !detail) return;

      var peers = statsData ? (statsData.peers || 0) : 0;
      var hostedShards = statsData ? (statsData.hosted_shards || 0) : 0;
      if (hostedShards === 0) {
        var el = document.getElementById('hosted-shards');
        if (el) hostedShards = parseInt(el.textContent, 10) || 0;
      }
      var hasLocalModel = hostedShards > 0;

      var cloudCount = 0;
      var cloudDown = 0;
      var seen = {};
      if (providerData && providerData.providers) {
        providerData.providers.forEach(function(p) {
          if (!p.configured) return;
          seen[p.name] = true;
          var h = S.providerHealth[p.name] || S.providerHealth[p.provider];
          var isHealthy = !h || h.status === 'up' || h.status === 'rate_limited' || h.status === 'overloaded';
          if (isHealthy) cloudCount++;
          else cloudDown++;
        });
      }
      Object.keys(S.providerHealth).forEach(function(key) {
        if (seen[key]) return;
        var h = S.providerHealth[key];
        var isHealthy = h.status === 'up' || h.status === 'rate_limited' || h.status === 'overloaded';
        if (isHealthy) cloudCount++;
        else cloudDown++;
      });

      if (indicator) indicator.className = 'mode-indicator mb-2';

      var modeName, dotClass, modeClass, modeHelp;

      if (peers > 0 && hasLocalModel && cloudCount > 0) {
        modeName = 'SWARM \u00b7 CLOUD'; dotClass = 'swarm'; modeClass = 'mode-hybrid'; modeHelp = 'Full power — swarm inference with cloud fallback';
      } else if (peers > 0 && hasLocalModel) {
        modeName = 'SWARM'; dotClass = 'swarm'; modeClass = 'mode-swarm'; modeHelp = 'Running inference locally and with peers';
      } else if (peers > 0) {
        modeName = 'SWARM \u00b7 REMOTE'; dotClass = 'swarm'; modeClass = 'mode-swarm'; modeHelp = 'Using peer nodes for inference (no local model)';
      } else if (hasLocalModel && cloudCount > 0) {
        modeName = 'LOCAL \u00b7 CLOUD'; dotClass = 'hybrid'; modeClass = 'mode-hybrid'; modeHelp = 'Local inference with cloud fallback';
      } else if (hasLocalModel) {
        modeName = 'SOLO'; dotClass = 'offline'; modeClass = 'mode-offline'; modeHelp = 'Local inference only — connect peers to unlock bigger models';
      } else if (cloudCount > 0) {
        modeName = 'CLOUD'; dotClass = 'cloud'; modeClass = 'mode-cloud'; modeHelp = 'Using cloud providers — download models for free local AI';
      } else {
        modeName = 'OFFLINE'; dotClass = 'offline'; modeClass = 'mode-offline'; modeHelp = 'Download a model or add a cloud provider to get started';
      }

      dot.className = 'mode-dot ' + dotClass;
      label.textContent = modeName;
      label.title = modeHelp;
      if (indicator) indicator.classList.add(modeClass);

      var requests = statsData ? (statsData.requests_made || 0) : 0;
      var served = statsData ? (statsData.served || 0) : 0;
      var active = statsData ? (statsData.active_requests || 0) : 0;

      var parts = [];
      if (peers > 0) parts.push('<span class="mode-stat"><strong>' + peers + '</strong> peer' + (peers !== 1 ? 's' : '') + '</span>');
      if (hostedShards > 0) parts.push('<span class="mode-stat"><strong>' + hostedShards + '</strong> shard' + (hostedShards !== 1 ? 's' : '') + '</span>');
      if (cloudCount > 0) parts.push('<span class="mode-stat"><strong>' + cloudCount + '</strong> provider' + (cloudCount !== 1 ? 's' : '') + '</span>');
      if (active > 0) parts.push('<span class="mode-stat" style="color:var(--orange)"><strong>' + active + '</strong> active</span>');
      if (requests > 0) parts.push('<span class="mode-stat"><strong>' + requests + '</strong> req</span>');
      if (served > 0) parts.push('<span class="mode-stat"><strong>' + served + '</strong> served</span>');

      var detailHtml;
      if (parts.length > 0) {
        detailHtml = parts.join('<span class="mode-separator">\u00b7</span>');
      } else {
        detailHtml = '<span class="mode-action" data-goto-hf="1">Connect to peers to access models, or add a cloud provider for instant chat</span>';
      }
      detail.innerHTML = detailHtml;
    },

    load: async function() {
      var statsData = App.data.cache.stats;
      if (!statsData) {
        try {
          var resp = await App.authFetch('/api/admin/stats');
          if (resp.ok) statsData = await resp.json();
        } catch (e) {}
      }
      var providerData = null;
      try {
        var resp2 = await App.authFetch('/api/admin/providers');
        if (resp2.ok) providerData = await resp2.json();
        S._cachedProviderData = providerData;
      } catch (e) {}
      App.modeIndicator.update(statsData, providerData);
    }
  };
})();
