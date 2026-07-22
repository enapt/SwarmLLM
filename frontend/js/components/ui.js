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
          : tab === 'responses' ? '/admin/responses'
          : tab === 'devices' ? '/admin/devices'
          : tab === 'swarm' ? '/admin/swarm'
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
      var responsesView = document.getElementById('view-responses');
      if (responsesView) responsesView.style.display = tab === 'responses' ? '' : 'none';
      var devicesView = document.getElementById('view-devices');
      if (devicesView) devicesView.style.display = tab === 'devices' ? '' : 'none';
      var swarmView = document.getElementById('view-swarm');
      if (swarmView) swarmView.style.display = tab === 'swarm' ? '' : 'none';
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
      if (tab === 'responses' && App.responses) {
        App.responses.enter();
      } else if (App.responses) {
        App.responses.leave();
      }
      if (tab === 'devices' && App.pool) {
        App.pool.load();
      }
      if (tab === 'swarm' && App.swarmTab) {
        App.swarmTab.onShow();
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

    // Jumps to the Models tab's Search subtab. The legacy modal has been
    // deleted — closeModelBrowser is no longer needed.
    openModelBrowser: function(query) {
      if (App.swarmTab && typeof App.swarmTab.openSearch === 'function') {
        App.swarmTab.openSearch(query || '');
        return;
      }
      App.ui.switchTab('swarm');
    },

    showBanner: function(type, message) {
      App.notifications.showToast(message, type === 'warning' ? 'warning' : type === 'error' ? 'error' : type === 'success' ? 'success' : 'info');
    }
  };

  // --- Network Status panel ---
  // Single source of truth for "what state is this node in?". Replaces the
  // old swarm-capacity-banner + mode-indicator strip. Six named states:
  //   connecting | global | private | lan | solo | offline
  // Cloud-provider count + capacity facts fold in as supporting detail.
  App.networkStatus = {
    update: function(statsData, providerData) {
      var panel = document.getElementById('network-status-panel');
      if (!panel) return;
      var dotEl = document.getElementById('netstatus-dot');
      var nameEl = document.getElementById('netstatus-name');
      var chipsEl = document.getElementById('netstatus-chips');
      var detailEl = document.getElementById('netstatus-detail');
      var modelsEl = document.getElementById('netstatus-models');

      var stats = statsData || null;
      var peers = stats ? (stats.peers || 0) : 0;
      var lanPeers = stats ? (stats.lan_peers || 0) : 0;
      // Mutually-exclusive peer taxonomy from the backend (Pool > LAN > Remote).
      // remotePeers + lanPeers + poolPeers === peers.
      var poolPeers = stats ? (stats.pool_peers || 0) : 0;
      var remotePeers = stats ? (stats.remote_peers || 0) : 0;
      var hostedShards = stats ? (stats.hosted_shards || 0) : 0;
      if (hostedShards === 0) {
        var hsEl = document.getElementById('hosted-shards');
        if (hsEl) hostedShards = parseInt(hsEl.textContent, 10) || 0;
      }
      var hasLocalModel = hostedShards > 0;
      var netMode = (stats && stats.network_mode) || {};
      var privateMode = !!netMode.private_mode;
      var offlineMode = !!netMode.offline_mode;
      var allowLan = !!netMode.private_mode_allow_lan;
      var capacity = stats ? stats.swarm_capacity : null;

      // Healthy cloud provider count (configured + reachable).
      var cloudCount = 0;
      var seen = {};
      if (providerData && providerData.providers) {
        providerData.providers.forEach(function(p) {
          if (!p.configured) return;
          seen[p.name] = true;
          var h = S.providerHealth[p.name] || S.providerHealth[p.provider];
          var isHealthy = !h || h.status === 'up' || h.status === 'rate_limited' || h.status === 'overloaded';
          if (isHealthy) cloudCount++;
        });
      }
      Object.keys(S.providerHealth).forEach(function(key) {
        if (seen[key]) return;
        var h = S.providerHealth[key];
        var isHealthy = h.status === 'up' || h.status === 'rate_limited' || h.status === 'overloaded';
        if (isHealthy) cloudCount++;
      });
      if (providerData && providerData.claude_subscription && providerData.claude_subscription.enabled) {
        cloudCount++;
      }

      // Pick the named state. Private/offline modes override "global"
      // because they describe an intentional restriction on outbound
      // inference scope (src/pool/scope.rs).
      var stateKey, stateName, stateDetail;
      if (!stats) {
        stateKey = 'connecting';
        stateName = I18n.t('netstatus.connecting');
        stateDetail = I18n.t('netstatus.detail_connecting');
      } else if (privateMode && offlineMode) {
        stateKey = 'lan';
        stateName = I18n.t('netstatus.lan');
        stateDetail = I18n.t('netstatus.detail_lan');
      } else if (privateMode) {
        stateKey = 'private';
        stateName = I18n.t('netstatus.private');
        stateDetail = allowLan
          ? I18n.t('netstatus.detail_private_with_lan')
          : I18n.t('netstatus.detail_private');
      } else if (peers > 0) {
        stateKey = 'global';
        stateName = I18n.t('netstatus.global');
        stateDetail = hasLocalModel
          ? I18n.t('netstatus.detail_global_hosting')
          : I18n.t('netstatus.detail_global_remote');
      } else if (hasLocalModel || cloudCount > 0) {
        stateKey = 'solo';
        stateName = I18n.t('netstatus.solo');
        stateDetail = hasLocalModel
          ? I18n.t('netstatus.detail_solo_local')
          : I18n.t('netstatus.detail_solo_cloud');
      } else {
        stateKey = 'offline';
        stateName = I18n.t('netstatus.offline');
        stateDetail = I18n.t('netstatus.detail_offline');
      }

      panel.className = 'panel network-status network-status-' + stateKey;
      if (nameEl) nameEl.textContent = stateName;
      if (dotEl) dotEl.title = stateDetail || '';

      if (chipsEl) {
        chipsEl.innerHTML = '';
        var addChip = function(html) {
          var c = document.createElement('span');
          c.className = 'netstatus-chip';
          c.innerHTML = html;
          chipsEl.appendChild(c);
        };
        if (stateKey === 'global' && peers > 0) {
          // One chip per non-empty peer type so the header spells out exactly
          // who's connected: internet peers, same-network peers, pool devices.
          if (remotePeers > 0) addChip('<strong>' + remotePeers + '</strong> ' + U.escapeHtml(I18n.t(remotePeers === 1 ? 'netstatus.chip_remote_one' : 'netstatus.chip_remote_other')));
          if (lanPeers > 0) addChip('<strong>' + lanPeers + '</strong> ' + U.escapeHtml(I18n.t(lanPeers === 1 ? 'netstatus.chip_lan_peer_one' : 'netstatus.chip_lan_peer_other')));
          if (poolPeers > 0) addChip('<strong>' + poolPeers + '</strong> ' + U.escapeHtml(I18n.t(poolPeers === 1 ? 'netstatus.chip_pool_one' : 'netstatus.chip_pool_other')));
        } else if (stateKey === 'lan' && lanPeers > 0) {
          addChip('<strong>' + lanPeers + '</strong> ' + U.escapeHtml(I18n.t(lanPeers === 1 ? 'netstatus.chip_lan_peer_one' : 'netstatus.chip_lan_peer_other')));
        } else if (stateKey === 'private') {
          if (poolPeers > 0) {
            addChip('<strong>' + poolPeers + '</strong> ' + U.escapeHtml(I18n.t(poolPeers === 1 ? 'netstatus.chip_pool_one' : 'netstatus.chip_pool_other')));
          } else {
            addChip(U.escapeHtml(I18n.t('netstatus.chip_pool_only')));
          }
          if (allowLan && lanPeers > 0) addChip('<strong>' + lanPeers + '</strong> ' + U.escapeHtml(I18n.t(lanPeers === 1 ? 'netstatus.chip_lan_peer_one' : 'netstatus.chip_lan_peer_other')));
        } else if (stateKey === 'solo') {
          // Don't echo a "local parts" chip — the supporting models line
          // below already names them. Cloud provider count is the only
          // useful chip here.
          if (cloudCount > 0) addChip('<strong>' + cloudCount + '</strong> ' + U.escapeHtml(I18n.t(cloudCount === 1 ? 'netstatus.chip_provider_one' : 'netstatus.chip_provider_other')));
        }
      }

      if (detailEl) {
        var detailHtml = U.escapeHtml(stateDetail || '');
        if (stateKey === 'global' && cloudCount > 0) {
          detailHtml += ' · ' + U.escapeHtml(I18n.t(cloudCount === 1 ? 'netstatus.also_cloud_one' : 'netstatus.also_cloud_other', { count: cloudCount }));
        }
        detailEl.innerHTML = detailHtml;
      }

      if (modelsEl) {
        modelsEl.innerHTML = '';
        modelsEl.style.display = 'none';
        if (capacity && stateKey !== 'connecting' && stateKey !== 'offline') {
          var modelsParts = [];
          var serveable = capacity.serveable_models || [];
          var vramMb = capacity.total_vram_mb || 0;
          // Below 1 GB reads as noise on a swarm-wide total, so it stays
          // blank rather than rendering "< 1 MB" — the empty string is what
          // the guard below tests. Formatting itself comes from formatSize.
          var memText = vramMb >= 1024 ? U.formatSize(vramMb) : '';
          if (memText && (peers > 0 || privateMode)) {
            modelsParts.push('<strong>' + memText + '</strong> ' + U.escapeHtml(I18n.t('netstatus.memory_word')));
          }
          if (serveable.length > 0) {
            var names = serveable.slice(0, 3).map(function(m) {
              var d = m.display_name || '';
              var looksRaw = d && d === d.toLowerCase() && /_/.test(d);
              if (d && !looksRaw) return U.escapeHtml(d);
              var src = d || m.model_id || '';
              return U.escapeHtml(U.formatModelDisplayName ? U.formatModelDisplayName(src) : src);
            }).join(', ');
            var more = serveable.length > 3 ? ' +' + (serveable.length - 3) : '';
            modelsParts.push(U.escapeHtml(I18n.t('netstatus.runs')) + ' ' + names + more);
          } else if (hasLocalModel) {
            modelsParts.push('<strong>' + hostedShards + '</strong> ' + U.escapeHtml(I18n.t(hostedShards === 1 ? 'netstatus.shard_one' : 'netstatus.shard_other')));
          }
          if (modelsParts.length > 0) {
            modelsEl.innerHTML = modelsParts.join(' · ');
            modelsEl.style.display = '';
          }
        }
      }

      // Swarm resources strip: spell out the collective hardware the swarm has
      // right now (computers online incl. yours, GPU machines, combined VRAM,
      // shared storage, regions). This is the "how big is the swarm actually?"
      // answer the header was missing. online_nodes already counts self + peers.
      var resourcesEl = document.getElementById('netstatus-resources');
      if (resourcesEl) {
        resourcesEl.innerHTML = '';
        resourcesEl.style.display = 'none';
        var showRes = capacity && stateKey !== 'connecting' && stateKey !== 'offline' &&
          (capacity.online_nodes || 0) > 1 && (peers > 0 || privateMode);
        if (showRes) {
          var fmtGB = function(mb, dp) {
            if (mb >= 1024 * 1024) return (mb / (1024 * 1024)).toFixed(2) + ' TB';
            return (mb / 1024).toFixed(dp) + ' GB';
          };
          var resParts = [];
          var nodes = capacity.online_nodes || 0;
          resParts.push('<strong>' + nodes + '</strong> ' + U.escapeHtml(I18n.t(nodes === 1 ? 'netstatus.res_computers_one' : 'netstatus.res_computers_other')));
          var gpus = capacity.gpu_nodes || 0;
          if (gpus > 0) resParts.push('<strong>' + gpus + '</strong> ' + U.escapeHtml(I18n.t(gpus === 1 ? 'netstatus.res_gpu_one' : 'netstatus.res_gpu_other')));
          var vmb = capacity.total_vram_mb || 0;
          if (vmb > 0) resParts.push('<strong>' + fmtGB(vmb, 1) + '</strong> ' + U.escapeHtml(I18n.t('netstatus.memory_word')));
          var dmb = capacity.total_disk_mb || 0;
          if (dmb > 0) resParts.push('<strong>' + fmtGB(dmb, 0) + '</strong> ' + U.escapeHtml(I18n.t('netstatus.res_disk')));
          var regions = capacity.regions_represented || 0;
          if (regions > 1) resParts.push('<strong>' + regions + '</strong> ' + U.escapeHtml(I18n.t('netstatus.res_regions_other')));
          resourcesEl.innerHTML = '<span class="netstatus-res-icon" aria-hidden="true">🌐</span> ' + resParts.join(' · ');
          resourcesEl.style.display = '';
        }
      }
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
      App.networkStatus.update(statsData, providerData);
      App.networkStatus.updateClaudeCodeBadge(providerData);
    },

    updateClaudeCodeBadge: function(providerData) {
      var phCC = document.getElementById('ph-claude-code');
      var sub = providerData && providerData.claude_subscription;
      if (!sub || !sub.enabled) {
        if (phCC) phCC.classList.add('hidden');
        return;
      }

      if (phCC) {
        phCC.classList.remove('hidden');
        var healthBar = document.getElementById('provider-health-bar');
        if (healthBar) healthBar.classList.remove('hidden');
      }

      var phPlanEl = document.getElementById('ph-cc-plan');
      if (phPlanEl && !phPlanEl.textContent) {
        App.data.loadClaudeSubStatus().then(function(data) {
          if (!data) return;
          var label = '';
          if (data.subscription_type) {
            var typeName = data.subscription_type.charAt(0).toUpperCase() + data.subscription_type.slice(1);
            label = I18n.t('settings.subscription_plan_label', { type: typeName });
          }
          if (data.cli_version) {
            label = (label ? label + ' · ' : '') + data.cli_version.replace(' (Claude Code)', '');
          }
          if (phPlanEl) phPlanEl.textContent = label;
        }).catch(function() {});
      }
    }
  };
})();
