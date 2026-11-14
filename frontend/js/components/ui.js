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
      // The overflow trigger has no `data-tab`, so the loop above always
      // clears it. Light it while one of ITS tabs is showing — otherwise
      // choosing Leaderboard leaves a header where nothing is selected and
      // the reader cannot tell where they are. Closing the panel here rather
      // than in the click handler covers every way of arriving: the keyboard,
      // the browser's Back button, and a link from another view.
      var moreTrigger = document.getElementById('nav-more-trigger');
      var morePanel = document.getElementById('nav-more-panel');
      if (moreTrigger && morePanel) {
        var inMenu = Array.prototype.some.call(
          morePanel.querySelectorAll('.tab-btn[data-tab]'),
          function(b) { return b.dataset.tab === tab; }
        );
        moreTrigger.classList.toggle('active', inMenu);
        morePanel.style.display = 'none';
        moreTrigger.setAttribute('aria-expanded', 'false');
      }
      document.getElementById('view-chat').style.display = tab === 'chat' ? '' : 'none';
      document.getElementById('view-dashboard').style.display = tab === 'dashboard' ? '' : 'none';
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
        // Every other tab refreshes something on entry; chat refreshed
        // nothing, so an empty state built at page load — before the first
        // stats had arrived — was still on screen minutes later however much
        // the swarm had since told us (report #027). No-ops when a
        // conversation is showing.
        if (App.chat.refreshEmptyState) App.chat.refreshEmptyState();
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
      // One destination, two panels: the map and the leaderboard are the
      // same question asked twice, so entering refreshes both.
      if (tab === 'network-map') {
        App.networkMap.refresh();
        if (App.identity && App.identity.loadLeaderboard) App.identity.loadLeaderboard();
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
      var modal = document.getElementById('settings-modal');
      // Already open: do not re-run `load`, which would repopulate the form
      // from the node and throw away what the user has typed.
      var wasOpen = !modal.classList.contains('hidden');
      modal.classList.remove('hidden');
      if (!wasOpen) {
        App.settings.markClean();
        App.settings.load();
      }
      if (scrollToProviders) {
        var section = document.getElementById('settings-providers-section');
        if (section) {
          section.open = true;
          setTimeout(function() { section.scrollIntoView({ behavior: 'smooth', block: 'center' }); }, 100);
        }
      }
    },

    // THE one way Settings closes: the close button, Escape, a click on the
    // backdrop, "Run setup again" and a finished Save all call this, so the
    // unsaved-changes guard cannot be skipped by one of them. Returns true when
    // the panel is now closed, false when the user chose to keep editing, so a
    // caller that does something AFTER closing (re-running setup) can stop.
    //
    // Prior art: every close request goes through one handler that checks the
    // form's dirty state and asks "discard unsaved changes?" before closing
    // (the usual dirty-form modal pattern; MUI/Hyperscript examples). The
    // dashboard confirms with the native `confirm()` everywhere else, so this
    // does too.
    closeSettings: function() {
      var modal = document.getElementById('settings-modal');
      if (modal.classList.contains('hidden')) return true;
      if (App.settings.hasUnsavedChanges()) {
        if (!confirm(I18n.t('settings.discard_confirm'))) return false;
        App.settings.discardEdits();
      }
      modal.classList.add('hidden');
      return true;
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
  // old swarm-capacity-banner + mode-indicator strip. Nine named states:
  //   unreachable | connecting | global | behind | viewonly | private | lan | solo | offline
  // Cloud-provider count + capacity facts fold in as supporting detail.
  // A cause the backend names (`UpnpVerdict::key`) → its fix. `searching`
  // and `mapped` have none: the next reachability check settles them.
  var REACH_FIX = {
    no_router: 'netstatus.reach_fix_no_router',
    refused: 'netstatus.reach_fix_refused',
    behind_another_nat: 'netstatus.reach_fix_behind_nat',
    off: 'netstatus.reach_fix_off',
    wsl_default_networking: 'netstatus.reach_fix_wsl',
    wsl_mirrored: 'netstatus.reach_fix_wsl_mirrored'
  };
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
      var swarmUse = stats ? stats.swarm_use : null;

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
      if (!stats && !App.data.loadReachedDaemon('stats')) {
        // Nothing to show because the page could not ask, not because the
        // node is still finding computers: "Looking for other computers" is a
        // claim about a node the page has not heard from (#69).
        stateKey = 'unreachable';
        stateName = I18n.t('netstatus.unreachable');
        stateDetail = I18n.t('errors.server_unreachable');
      } else if (!stats) {
        stateKey = 'connecting';
        stateName = I18n.t('netstatus.connecting');
        stateDetail = I18n.t('netstatus.detail_connecting');
      } else if (privateMode && offlineMode) {
        stateKey = 'lan';
        stateName = I18n.t('netstatus.lan');
        stateDetail = I18n.t('netstatus.detail_lan');
      } else if (privateMode) {
        stateKey = 'private';
        // Private mode admits computers on the local network by default
        // (`private_mode_allow_lan`), so the headline names them when it does.
        stateName = allowLan
          ? I18n.t('netstatus.private_with_lan')
          : I18n.t('netstatus.private');
        stateDetail = allowLan
          ? I18n.t('netstatus.detail_private_with_lan')
          : I18n.t('netstatus.detail_private');
      } else if (swarmUse && swarmUse.verdict === 'too_far_behind') {
        // Two or more releases behind: computers on newer releases disconnect
        // this one (`swarm_use::cut_off_for`), so the panel says so whether or
        // not any computer is still connected — with none, "Solo" or
        // "Offline" would hide the reason and the fix.
        stateKey = 'behind';
        stateName = I18n.t('netstatus.behind');
        stateDetail = I18n.t('netstatus.detail_behind', { running: swarmUse.running, newest: swarmUse.newest });
      } else if (peers > 0 && swarmUse && swarmUse.verdict && swarmUse.verdict !== 'allowed') {
        // Connected, but other computers will not run this node's chats
        // (`daemon::state::swarm_use`): it can look at the swarm, not use it.
        // After the private branches — there only pool devices are asked.
        stateKey = 'viewonly';
        stateName = I18n.t('netstatus.viewonly');
        stateDetail = I18n.t('netstatus.detail_viewonly_parts', { held: swarmUse.parts_held, needed: swarmUse.parts_needed });
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
        // A bold count and its label (a plural key's `count` picks
        // `_one` / `_other`, the n === 1 rule).
        var addCountChip = function(n, label) {
          addChip('<strong>' + n + '</strong> ' + U.escapeHtml(label));
        };
        if ((stateKey === 'global' || stateKey === 'viewonly' || stateKey === 'behind') && peers > 0) {
          // One chip per non-empty peer type so the header spells out exactly
          // who's connected: internet peers, same-network peers, pool devices.
          if (remotePeers > 0) addCountChip(remotePeers, I18n.t('netstatus.chip_remote', { count: remotePeers }));
          if (lanPeers > 0) addCountChip(lanPeers, I18n.t('netstatus.chip_lan_peer', { count: lanPeers }));
          if (poolPeers > 0) addCountChip(poolPeers, I18n.t('netstatus.chip_pool', { count: poolPeers }));
        } else if (stateKey === 'lan' && lanPeers > 0) {
          addCountChip(lanPeers, I18n.t('netstatus.chip_lan_peer', { count: lanPeers }));
        } else if (stateKey === 'private') {
          if (poolPeers > 0) {
            addCountChip(poolPeers, I18n.t('netstatus.chip_pool', { count: poolPeers }));
          } else {
            addChip(U.escapeHtml(I18n.t('netstatus.chip_pool_only')));
          }
          if (allowLan && lanPeers > 0) addCountChip(lanPeers, I18n.t('netstatus.chip_lan_peer', { count: lanPeers }));
        } else if (stateKey === 'solo') {
          // Don't echo a "local parts" chip — the supporting models line
          // below already names them. Cloud provider count is the only
          // useful chip here.
          if (cloudCount > 0) addCountChip(cloudCount, I18n.t('netstatus.chip_provider', { count: cloudCount }));
        }
      }

      if (detailEl) {
        var detailHtml = U.escapeHtml(stateDetail || '');
        if (stateKey === 'global' && cloudCount > 0) {
          detailHtml += ' · ' + U.escapeHtml(I18n.t(cloudCount === 1 ? 'netstatus.also_cloud_one' : 'netstatus.also_cloud_other', { count: cloudCount }));
        }
        detailEl.innerHTML = detailHtml;
      }

      // Reached only through a relay, which is slower: name what is in the
      // way and its one fix (`SharedState::reachability_json`, the backend's
      // `UpnpVerdict`), in the order Xbox's NAT test and Plex's Remote Access
      // advise. Nothing while UPnP is still asking or holds a mapping.
      var reachEl = document.getElementById('netstatus-reach');
      if (reachEl) {
        var reach = stats ? stats.reachability : null;
        var fixKey = reach && reach.direct === false ? REACH_FIX[reach.upnp] : null;
        if (fixKey && (stateKey === 'global' || stateKey === 'viewonly')) {
          var fix = I18n.t(fixKey, { udp: reach.udp_port, tcp: reach.tcp_port, other: reach.udp_port <= 65000 ? reach.udp_port + 100 : reach.udp_port - 100 });
          reachEl.innerHTML = '<span class="netstatus-reach-icon" aria-hidden="true">⚠</span> <span>'
            + U.escapeHtml(I18n.t('netstatus.reach_relay') + ' ' + fix) + '</span>';
          reachEl.style.display = '';
        } else {
          reachEl.innerHTML = '';
          reachEl.style.display = 'none';
        }
      }

      if (modelsEl) {
        modelsEl.innerHTML = '';
        modelsEl.style.display = 'none';
        if (capacity && stateKey !== 'connecting' && stateKey !== 'offline') {
          var modelsParts = [];
          var serveable = capacity.serveable_models || [];
          // Graphics memory is NOT repeated here. It is the same
          // `capacity.total_vram_mb` the resources strip renders two lines
          // below (labelled "graphics memory" since a tester read "combined
          // memory" as everyone's RAM, field report 2026-10-15), so a reader met
          // one number twice and had to work out they were the same fact —
          // on a live node: "13.7 GB combined memory · Runs …" directly above
          // "7 computers · 2 with GPUs · 13.7 GB combined memory · …".
          // This line answers what the swarm can RUN; the strip answers what
          // hardware it HAS, and memory is a hardware fact.
          if (serveable.length > 0) {
            // Named from the id, like every other list of models: two
            // quantisations share the file's own name, and the banner read
            // "Runs Qwen3-30B-A3B-Instruct-2507, Qwen3-30B-A3B-Instruct-2507"
            // (field report 2026-10-15).
            var names = serveable.slice(0, 3).map(function(m) {
              return U.escapeHtml(U.formatModelDisplayName(m.model_id || m.display_name || ''));
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
      // A failed load answers with the providers last read, so a hiccup no
      // longer drops every cloud provider from the count, and with them the
      // headline (to "Offline" when they were all this node had) and the
      // Claude Code badge (#69).
      var providerData = null;
      try {
        providerData = await App.data.loadProviders();
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
            var typeName = U.capitalize(data.subscription_type);
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
