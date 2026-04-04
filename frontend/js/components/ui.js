'use strict';

// ============================================================================
// SwarmLLM — UI Module
// Tab switching, sidebar, modals, mode indicator
// ============================================================================

(function() {
  var S = App.state;
  var U = App.utils;

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
      if (tab === 'dashboard') {
        // Only reload if data is older than 10s to avoid rate limit hammering
        var lastLoad = App.dashboard._lastLoadTime || 0;
        if (Date.now() - lastLoad > 10000) {
          App.dashboard.loadInitial();
        }
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
      var claudeSubEnabled = false;
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
      // Claude subscription counts as a provider
      if (providerData && providerData.claude_subscription && providerData.claude_subscription.enabled) {
        claudeSubEnabled = true;
        cloudCount++;
      }

      if (indicator) indicator.className = 'mode-indicator mb-2';

      var modeName, dotClass, modeClass, modeHelp;

      if (peers > 0 && hasLocalModel && cloudCount > 0) {
        modeName = I18n.t('mode.swarm_cloud'); dotClass = 'swarm'; modeClass = 'mode-hybrid'; modeHelp = I18n.t('mode.help_swarm_cloud');
      } else if (peers > 0 && hasLocalModel) {
        modeName = I18n.t('mode.swarm'); dotClass = 'swarm'; modeClass = 'mode-swarm'; modeHelp = I18n.t('mode.help_swarm');
      } else if (peers > 0) {
        modeName = I18n.t('mode.swarm_remote'); dotClass = 'swarm'; modeClass = 'mode-swarm'; modeHelp = I18n.t('mode.help_swarm_remote');
      } else if (hasLocalModel && cloudCount > 0) {
        modeName = I18n.t('mode.local_cloud'); dotClass = 'hybrid'; modeClass = 'mode-hybrid'; modeHelp = I18n.t('mode.help_local_cloud');
      } else if (hasLocalModel) {
        modeName = I18n.t('mode.solo'); dotClass = 'offline'; modeClass = 'mode-offline'; modeHelp = I18n.t('mode.help_solo');
      } else if (cloudCount > 0) {
        modeName = I18n.t('mode.cloud'); dotClass = 'cloud'; modeClass = 'mode-cloud'; modeHelp = I18n.t('mode.help_cloud');
      } else {
        modeName = I18n.t('mode.offline'); dotClass = 'offline'; modeClass = 'mode-offline'; modeHelp = I18n.t('mode.help_offline');
      }

      dot.className = 'mode-dot ' + dotClass;
      label.textContent = modeName;
      label.title = modeHelp;
      if (indicator) indicator.classList.add(modeClass);

      var requests = statsData ? (statsData.requests_made || 0) : 0;
      var served = statsData ? (statsData.served || 0) : 0;
      var active = statsData ? (statsData.active_requests || 0) : 0;

      var parts = [];
      if (peers > 0) parts.push('<span class="mode-stat"><strong>' + peers + '</strong> ' + I18n.t(peers !== 1 ? 'mode.stat_peers_other' : 'mode.stat_peers_one', { count: peers }).replace(/^\d+\s*/, '') + '</span>');
      if (hostedShards > 0) parts.push('<span class="mode-stat"><strong>' + hostedShards + '</strong> ' + I18n.t(hostedShards !== 1 ? 'mode.stat_shards_other' : 'mode.stat_shards_one', { count: hostedShards }).replace(/^\d+\s*/, '') + '</span>');
      if (cloudCount > 0) parts.push('<span class="mode-stat"><strong>' + cloudCount + '</strong> ' + I18n.t(cloudCount !== 1 ? 'mode.stat_providers_other' : 'mode.stat_providers_one', { count: cloudCount }).replace(/^\d+\s*/, '') + '</span>');
      if (active > 0) parts.push('<span class="mode-stat" style="color:var(--orange)"><strong>' + active + '</strong> ' + I18n.t('mode.stat_active', { count: active }).replace(/^\d+\s*/, '') + '</span>');
      if (requests > 0) parts.push('<span class="mode-stat"><strong>' + requests + '</strong> ' + I18n.t('mode.stat_requests', { count: requests }).replace(/^\d+\s*/, '') + '</span>');
      if (served > 0) parts.push('<span class="mode-stat"><strong>' + served + '</strong> ' + I18n.t('mode.stat_served', { count: served }).replace(/^\d+\s*/, '') + '</span>');

      // Claude Code badge when subscription active
      if (claudeSubEnabled) {
        parts.push('<span class="mode-stat mode-claude-badge">' + providerIconHtml('claude_subscription', 13) + ' <strong>Claude Code</strong></span>');
      }

      var detailHtml;
      if (parts.length > 0) {
        detailHtml = parts.join('<span class="mode-separator">\u00b7</span>');
      } else {
        detailHtml = '<span class="mode-action" data-goto-hf="1">' + U.escapeHtml(I18n.t('mode.empty_cta')) + '</span>';
      }
      detail.innerHTML = detailHtml;
    },

    load: async function() {
      var statsData = App.data.cache.stats;
      if (!statsData) {
        try {
          var result = await App.data.loadStats();
          statsData = result.stats;
        } catch (e) {}
      }
      var providerData = null;
      try {
        providerData = await App.data.loadProviders();
        S._cachedProviderData = providerData;
      } catch (e) {}
      App.modeIndicator.update(statsData, providerData);
    }
  };
})();
