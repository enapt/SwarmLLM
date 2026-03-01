'use strict';

// ============================================================================
// SwarmLLM — Unified single-page application
// ============================================================================

var SwarmLLM = (function() {
  var ws = null;
  var wsHealthy = false;
  var pollTimers = [];
  var creditHistory = [];
  var activeAcquisitions = {};
  var isStreaming = false;
  var currentModel = '';
  var currentSessionId = null;
  var sessions = {};
  var activeTab = 'dashboard';

  // --- STORAGE KEYS ---
  var SESSIONS_KEY = 'swarmllm_sessions';
  var ACTIVE_SESSION_KEY = 'swarmllm_active_session';
  var SETUP_DONE_KEY = 'swarmllm_setup_done';

  // --- HELPERS ---
  function escapeHtml(str) {
    var div = document.createElement('div');
    div.textContent = str || '';
    return div.innerHTML;
  }

  // Authenticated fetch — adds Bearer token to all requests that need auth
  function authFetch(url, opts) {
    opts = opts || {};
    opts.headers = opts.headers || {};
    if (settings._apiKeyFull && !opts.headers['Authorization']) {
      opts.headers['Authorization'] = 'Bearer ' + settings._apiKeyFull;
    }
    return fetch(url, opts);
  }

  // ========================================================================
  // UI Module — tab switching, sidebar, modals
  // ========================================================================
  var ui = {
    switchTab: function(tab) {
      activeTab = tab;
      document.querySelectorAll('.tab-btn').forEach(function(b) {
        b.classList.toggle('active', b.dataset.tab === tab);
      });
      document.getElementById('view-chat').style.display = tab === 'chat' ? '' : 'none';
      document.getElementById('view-dashboard').style.display = tab === 'dashboard' ? '' : 'none';
      var lbView = document.getElementById('view-leaderboard');
      if (lbView) lbView.style.display = tab === 'leaderboard' ? '' : 'none';
      var mapView = document.getElementById('view-network-map');
      if (mapView) mapView.style.display = tab === 'network-map' ? '' : 'none';
      if (tab === 'chat') {
        chat.scrollToBottom();
        document.getElementById('chat-input').focus();
      }
      if (tab === 'leaderboard') {
        identity.loadLeaderboard();
      }
      if (tab === 'network-map') {
        networkMap.refresh();
      }
    },

    toggleSidebar: function() {
      var sidebar = document.getElementById('sidebar');
      sidebar.classList.toggle('collapsed');
      var btn = sidebar.querySelector('.sidebar-toggle');
      btn.innerHTML = sidebar.classList.contains('collapsed') ? '&#9654;' : '&#9664;';
    },

    toggleMobileSidebar: function() {
      document.body.classList.toggle('sidebar-open');
    },

    openSettings: function() {
      document.getElementById('settings-modal').classList.remove('hidden');
      settings.load();
    },

    closeSettings: function() {
      document.getElementById('settings-modal').classList.add('hidden');
    },

    openModelBrowser: function() {
      document.getElementById('model-browser-modal').classList.remove('hidden');
    },

    closeModelBrowser: function() {
      document.getElementById('model-browser-modal').classList.add('hidden');
    },

    showBanner: function(type, message) {
      var banner = document.getElementById('status-banner');
      if (!banner) return;
      banner.innerHTML = '<div class="alert alert-' + type + '">' + escapeHtml(message) + '</div>';
      if (type === 'success') {
        setTimeout(function() { banner.innerHTML = ''; }, 3000);
      }
    }
  };

  // ========================================================================
  // Chat Module — sessions, messages, streaming
  // ========================================================================
  var chat = {
    handleKey: function(e) {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        chat.send();
      }
    },

    newSession: function() {
      var id = 'session_' + Date.now();
      sessions[id] = { id: id, title: 'New Chat', messages: [], created: Date.now() };
      currentSessionId = id;
      chat.saveSessions();
      chat.renderSessionList();
      chat.renderMessages();
      ui.switchTab('chat');
    },

    switchSession: function(id) {
      if (!sessions[id]) return;
      currentSessionId = id;
      localStorage.setItem(ACTIVE_SESSION_KEY, id);
      chat.renderSessionList();
      chat.renderMessages();
    },

    deleteSession: function(id, e) {
      if (e) { e.stopPropagation(); e.preventDefault(); }
      delete sessions[id];
      if (currentSessionId === id) {
        var keys = Object.keys(sessions);
        currentSessionId = keys.length > 0 ? keys[keys.length - 1] : null;
      }
      chat.saveSessions();
      chat.renderSessionList();
      chat.renderMessages();
    },

    renderSessionList: function() {
      var list = document.getElementById('session-list');
      var sorted = Object.values(sessions).sort(function(a, b) { return b.created - a.created; });
      if (sorted.length === 0) {
        list.innerHTML = '<div class="text-muted" style="padding:12px;font-size:0.8rem">No sessions yet</div>';
        return;
      }
      list.innerHTML = '';
      sorted.forEach(function(s) {
        var div = document.createElement('div');
        div.className = 'session-item' + (s.id === currentSessionId ? ' active' : '');
        div.onclick = function() { chat.switchSession(s.id); };
        var title = s.title.length > 28 ? s.title.substring(0, 28) + '...' : s.title;
        div.innerHTML = '<span class="session-title">' + escapeHtml(title) + '</span>' +
          '<button class="btn btn-ghost btn-sm session-delete" data-delete-session="' + escapeHtml(s.id) + '" title="Delete">&times;</button>';
        list.appendChild(div);
      });
    },

    renderMessages: function() {
      var container = document.getElementById('chat-messages');
      var empty = document.getElementById('chat-empty');
      container.innerHTML = '';

      if (!currentSessionId || !sessions[currentSessionId]) {
        container.appendChild(createEmptyState());
        return;
      }

      var msgs = sessions[currentSessionId].messages;
      if (msgs.length === 0) {
        container.appendChild(createEmptyState());
        return;
      }

      msgs.forEach(function(msg) {
        appendMessageToDOM(msg.role, msg.content);
      });
      chat.scrollToBottom();
    },

    send: async function() {
      if (isStreaming) return;

      var input = document.getElementById('chat-input');
      var text = input.value.trim();
      if (!text) return;

      // Ensure we have a session
      if (!currentSessionId || !sessions[currentSessionId]) {
        chat.newSession();
      }

      input.value = '';
      autoResizeInput();

      var session = sessions[currentSessionId];
      session.messages.push({ role: 'user', content: text });

      // Auto-title from first message
      if (session.messages.length === 1) {
        session.title = text.substring(0, 50);
        chat.renderSessionList();
      }

      chat.saveSessions();
      appendMessageToDOM('user', text);

      // Prepare assistant message for streaming
      var assistantEl = appendMessageToDOM('assistant', '');
      var contentEl = assistantEl.querySelector('.msg-content');
      contentEl.innerHTML = '<span class="typing-indicator">Thinking...</span>';

      isStreaming = true;
      document.getElementById('send-btn').disabled = true;
      var startTime = performance.now();

      var model = document.getElementById('model-select').value || currentModel || 'local';
      var body = {
        model: model,
        messages: session.messages.map(function(m) {
          return { role: m.role, content: m.content };
        }),
        temperature: 0.7,
        max_tokens: 2048,
        stream: true,
      };

      var fullContent = '';

      try {
        var resp = await authFetch('/v1/chat/completions', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body),
        });

        if (!resp.ok) {
          var errText = await resp.text();
          contentEl.textContent = 'Error: ' + errText;
          isStreaming = false;
          document.getElementById('send-btn').disabled = false;
          return;
        }

        var cleared = false;
        var reader = resp.body.getReader();
        var decoder = new TextDecoder();
        var buffer = '';

        while (true) {
          var result = await reader.read();
          if (result.done) break;

          buffer += decoder.decode(result.value, { stream: true });
          var lines = buffer.split('\n');
          buffer = lines.pop() || '';

          for (var i = 0; i < lines.length; i++) {
            var line = lines[i].trim();
            if (!line.startsWith('data:')) continue;
            var payload = line.substring(5).trim();
            if (payload === '[DONE]') continue;

            try {
              var chunk = JSON.parse(payload);
              if (chunk.choices && chunk.choices[0] && chunk.choices[0].delta) {
                var delta = chunk.choices[0].delta;
                if (delta.content) {
                  if (!cleared) { contentEl.textContent = ''; cleared = true; }
                  fullContent += delta.content;
                  contentEl.textContent = fullContent;
                  chat.scrollToBottom();
                }
              }
            } catch (e) {}
          }
        }

        if (!cleared && !fullContent) {
          contentEl.textContent = 'No response received. The model may still be loading.';
        }
      } catch (e) {
        if (!fullContent) {
          contentEl.textContent = 'Error: Connection failed.';
        }
      }

      // Show response time
      var elapsed = ((performance.now() - startTime) / 1000).toFixed(2);
      var timerEl = document.createElement('div');
      timerEl.className = 'msg-timer';
      timerEl.textContent = elapsed + 's';
      assistantEl.appendChild(timerEl);

      if (fullContent) {
        session.messages.push({ role: 'assistant', content: fullContent });
        chat.saveSessions();
      }

      isStreaming = false;
      document.getElementById('send-btn').disabled = false;
    },

    scrollToBottom: function() {
      var container = document.getElementById('chat-messages');
      container.scrollTop = container.scrollHeight;
    },

    saveSessions: function() {
      try {
        localStorage.setItem(SESSIONS_KEY, JSON.stringify(sessions));
        if (currentSessionId) localStorage.setItem(ACTIVE_SESSION_KEY, currentSessionId);
      } catch (e) {}
    },

    loadSessions: function() {
      try {
        var saved = localStorage.getItem(SESSIONS_KEY);
        if (saved) sessions = JSON.parse(saved);

        // Migrate old single-chat history
        var oldHistory = localStorage.getItem('swarmllm_chat_history');
        if (oldHistory && Object.keys(sessions).length === 0) {
          var msgs = JSON.parse(oldHistory);
          if (msgs.length > 0) {
            var id = 'session_migrated';
            sessions[id] = { id: id, title: msgs[0].content.substring(0, 50), messages: msgs, created: Date.now() - 1000 };
            localStorage.removeItem('swarmllm_chat_history');
          }
        }

        currentSessionId = localStorage.getItem(ACTIVE_SESSION_KEY);
        if (currentSessionId && !sessions[currentSessionId]) {
          currentSessionId = Object.keys(sessions).pop() || null;
        }
      } catch (e) {
        sessions = {};
      }
    }
  };

  // ========================================================================
  // Dashboard Module — stats, models, network
  // ========================================================================
  var dashboard = {
    loadInitial: async function() {
      try {
        var resp = await fetch('/api/admin/stats');
        var data = await resp.json();
        dashboard.updateFull(data);
      } catch (e) {
        ui.showBanner('error', 'Failed to connect to SwarmLLM daemon');
      }

      try {
        var resp = await fetch('/api/admin/config');
        var cfg = await resp.json();
        if (cfg.contribution) document.getElementById('settings-contribution').value = cfg.contribution;
        if (cfg.max_concurrent_requests) document.getElementById('settings-max-requests').value = cfg.max_concurrent_requests;
        if (cfg.max_bandwidth_mbps !== undefined) document.getElementById('settings-bandwidth').value = cfg.max_bandwidth_mbps;
        if (cfg.max_disk_mb) document.getElementById('settings-disk').value = cfg.max_disk_mb;
      } catch (e) {}

      try {
        var resp = await fetch('/api/admin/models');
        var models = await resp.json();
        dashboard.renderModels(models);
      } catch (e) {}

      dashboard.loadNetworkData();
      loadNetworkCode();
    },

    updateFull: function(data) {
      if (data.node_id) document.getElementById('node-id').textContent = data.node_id;
      if (data.version) document.getElementById('version').textContent = 'v' + data.version;
      if (data.uptime_seconds !== undefined) document.getElementById('uptime').textContent = formatUptime(data.uptime_seconds);
      if (data.tier) {
        setTierBadge('tier-badge', data.tier);
        setTierBadge('credit-tier', data.tier);
      }

      dashboard.updateStats(data);

      if (data.hardware) {
        var hw = data.hardware;
        if (hw.gpu_name) {
          document.getElementById('node-gpu').textContent = hw.gpu_name;
          if (hw.gpu_vram_mb) document.getElementById('node-vram').textContent = formatMB(hw.gpu_vram_mb) + ' VRAM';
        } else {
          document.getElementById('node-gpu').textContent = 'CPU only';
          document.getElementById('node-vram').textContent = '';
        }
        document.getElementById('node-cpu').textContent = hw.cpu_name ? hw.cpu_name + ' (' + hw.cpu_cores + ' cores)' : 'Unknown';

        if (hw.total_ram_mb) {
          document.getElementById('ram-total').textContent = '/ ' + formatMB(hw.total_ram_mb);
          var ramUsed = hw.used_ram_mb || 0;
          document.getElementById('ram-used').textContent = formatMB(ramUsed);
          var ramPct = hw.total_ram_mb > 0 ? (ramUsed / hw.total_ram_mb * 100) : 0;
          document.getElementById('ram-bar').style.width = ramPct.toFixed(1) + '%';
          document.getElementById('ram-bar').className = ramPct > 90 ? 'fill red' : (ramPct > 70 ? 'fill orange' : 'fill green');
        }
        if (hw.total_disk_mb) {
          document.getElementById('disk-total').textContent = '/ ' + formatMB(hw.total_disk_mb);
          var diskUsed = hw.used_disk_mb || 0;
          document.getElementById('disk-used').textContent = formatMB(diskUsed);
          var diskPct = hw.total_disk_mb > 0 ? (diskUsed / hw.total_disk_mb * 100) : 0;
          document.getElementById('disk-bar').style.width = diskPct.toFixed(1) + '%';
        }
      }

      if (data.hosted_shards !== undefined) document.getElementById('hosted-shards').textContent = data.hosted_shards;
    },

    updateStats: function(data) {
      if (data.peers !== undefined) document.getElementById('stat-peers').textContent = data.peers;
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
        // Track credit deltas for sparkline (shows rate of change, not absolute)
        var prevBal = creditHistory.length > 0 ? creditHistory[creditHistory.length - 1]._bal : bal;
        var delta = bal - prevBal;
        creditHistory.push({ _bal: bal, v: delta });
        if (creditHistory.length > 30) creditHistory.shift();
        renderSparkline('credit-sparkline', creditHistory.map(function(e) { return e.v; }));
      }
      if (data.requests_served !== undefined) document.getElementById('stat-served').textContent = data.requests_served;
      if (data.requests_made !== undefined) document.getElementById('stat-requests-made').textContent = data.requests_made;
      if (data.forwards_served !== undefined) document.getElementById('stat-forwards').textContent = data.forwards_served;
      if (data.active_requests !== undefined) document.getElementById('stat-active').textContent = data.active_requests;
    },

    renderModels: function(models) {
      var list = document.getElementById('models-list');
      var empty = document.getElementById('models-empty');

      if (!models || models.length === 0) {
        list.innerHTML = '';
        empty.style.display = '';
        return;
      }

      empty.style.display = 'none';
      list.innerHTML = '';

      models.forEach(function(m) {
        var shards = m.shards || [];
        var shardCount = m.shard_count || 0;
        var hostedShards = m.hosted_shards || 0;
        var globalAvail = m.global_available || hostedShards;
        var isDownloading = m.acquisition === 'downloading';
        var isReady = m.status === 'loaded' || m.status === 'ready' || (globalAvail === shardCount && shardCount > 0);
        var isPartial = !isReady && hostedShards > 0 && hostedShards < shardCount;
        var safeId = (m.id || '').replace(/[^a-zA-Z0-9]/g, '_');

        var card = document.createElement('div');
        card.className = 'model-card' + (isReady ? ' ready' : (isDownloading ? ' downloading' : (isPartial ? ' partial' : '')));
        card.setAttribute('data-model-id', m.id);

        // Status badge
        var statusHtml = '';
        if (m.status === 'loaded') {
          statusHtml = '<span style="color:var(--green);font-weight:600;font-size:0.8rem">Active</span>';
        } else if (isReady) {
          statusHtml = '<span style="color:var(--green);font-size:0.8rem">Ready (' + hostedShards + ' local, ' + globalAvail + '/' + shardCount + ' network)</span>';
        } else if (isDownloading) {
          statusHtml = '<span style="color:var(--accent);font-size:0.8rem"><span class="spinner" style="width:12px;height:12px;border-width:1.5px;margin-right:4px;vertical-align:middle"></span>Downloading</span>';
        } else if (isPartial) {
          statusHtml = '<span style="color:var(--orange);font-size:0.8rem">' + hostedShards + '/' + shardCount + ' local, ' + globalAvail + ' on network</span>';
        } else {
          statusHtml = '<span class="text-muted" style="font-size:0.8rem">Discovered</span>';
        }

        // Meta info
        var metaParts = [];
        metaParts.push(formatBytes(m.total_size_bytes || 0));
        if (shardCount > 1) metaParts.push(shardCount + ' shards');
        if (m.estimated_vram_mb) metaParts.push('~' + formatMB(m.estimated_vram_mb) + ' VRAM');
        if (m.peers_hosting > 0) metaParts.push(m.peers_hosting + ' peer' + (m.peers_hosting !== 1 ? 's' : ''));

        // Local file indicators (manifest + header needed to run shards)
        var fileIndicators = '';
        if (hostedShards > 0 || isDownloading) {
          var hasManifest = m.has_manifest !== false;
          var hasHeader = m.has_header !== false;
          if (!hasManifest || !hasHeader) {
            var missing = [];
            if (!hasManifest) missing.push('manifest');
            if (!hasHeader) missing.push('header');
            fileIndicators = '<span style="color:var(--orange);font-size:0.7rem;margin-left:6px" title="Missing: ' + missing.join(', ') + '">Missing ' + missing.join(' + ') + '</span>';
          }
        }

        // Shard grid
        var shardHtml = '';
        if (shardCount > 1 && shards.length > 0) {
          shardHtml = '<div class="shard-grid" data-model-grid="' + safeId + '">';
          var localCount = 0, peerCount = 0, dlCount = 0, peerDlCount = 0, queuedCount = 0, missingCount = 0;
          shards.forEach(function(s) {
            var cls = 'missing';
            var label = '' + s.index;
            var dlPct = 0;

            if (s.local) { cls = 'local'; localCount++; }
            else if (s.holders > 0) { cls = 'peer'; peerCount++; }
            else { missingCount++; }

            // Check per-shard download state (only marks THIS specific shard)
            if (s.download && s.download.state === 'Downloading') {
              dlPct = s.download.progress_pct || 0;
              cls = 'downloading'; dlCount++;
              if (missingCount > 0) missingCount--;
              if (peerCount > 0 && !s.local) peerCount--;
              label = dlPct + '%';
            }
            // Check if a peer is downloading this shard
            if (s.peer_downloads && s.peer_downloads.length > 0) {
              if (cls !== 'local' && cls !== 'downloading') {
                dlPct = s.peer_downloads[0].progress_pct || 0;
                cls = 'peer-downloading'; peerDlCount++;
                if (missingCount > 0) missingCount--;
                if (peerCount > 0) peerCount--;
                label = dlPct + '%';
              }
            }

            var title = 'Shard ' + s.index + ' (' + formatBytes(s.size_bytes) + ')';
            if (cls === 'local') title += ' \u2014 Stored locally';
            else if (cls === 'peer') title += ' \u2014 Available from ' + s.holders + ' peer(s)';
            else if (cls === 'downloading') title += ' \u2014 Downloading (' + dlPct + '%)';
            else if (cls === 'peer-downloading') title += ' \u2014 Peer downloading (' + dlPct + '%)';
            else title += ' \u2014 Not available';

            var style = '';
            if (cls === 'downloading' || cls === 'peer-downloading') {
              style = ' style="--dl-pct:' + dlPct + '%"';
            }
            shardHtml += '<div class="shard-cell ' + cls + '"' + style + ' data-shard="' + safeId + '-' + s.index + '" title="' + title + '">' + label + '</div>';
          });
          shardHtml += '</div>';

          // Legend
          var legendParts = [];
          if (localCount > 0) legendParts.push('<span class="leg-local">Local (' + localCount + ')</span>');
          if (peerCount > 0) legendParts.push('<span class="leg-peer">Peer (' + peerCount + ')</span>');
          if (dlCount > 0) legendParts.push('<span class="leg-dl">Downloading (' + dlCount + ')</span>');
          if (peerDlCount > 0) legendParts.push('<span class="leg-peer-dl">Peer DL (' + peerDlCount + ')</span>');
          if (queuedCount > 0) legendParts.push('<span class="leg-queued">Queued (' + queuedCount + ')</span>');
          if (missingCount > 0) legendParts.push('<span class="leg-missing">Missing (' + missingCount + ')</span>');
          if (legendParts.length > 0) shardHtml += '<div class="shard-legend" data-model-legend="' + safeId + '">' + legendParts.join('') + '</div>';
        }

        // Download progress bar — segmented by shard with ETA
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
            var secsLeft = Math.round((totalBytes - dlBytes) / speed);
            if (secsLeft >= 3600) etaStr = Math.floor(secsLeft / 3600) + 'h ' + Math.floor((secsLeft % 3600) / 60) + 'm';
            else if (secsLeft >= 60) etaStr = Math.floor(secsLeft / 60) + 'm ' + (secsLeft % 60) + 's';
            else etaStr = secsLeft + 's';
          }
          // Build segmented bar — one segment per downloading shard
          var dlShards = shards.filter(function(s) { return s.download || s.local; });
          var segmentCount = Math.max(dlShards.length, shardCount);
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
          var shardLabel = ap.downloaded_shards !== undefined ? ('Shard ' + ap.downloaded_shards + '/' + shardCount) : 'Downloading';
          var rightText = formatBytes(dlBytes) + ' / ' + formatBytes(totalBytes) + ' (' + pct + '%)';
          if (speed > 0) rightText += ' &middot; ' + formatSpeed(speed);
          if (etaStr) rightText += ' &middot; ETA ' + etaStr;
          progressHtml = '<div class="dl-progress" data-model-progress="' + safeId + '" data-last-pct="' + pct + '">' +
            '<div class="flex-between" style="font-size:0.75rem;margin-bottom:3px">' +
            '<span class="text-muted">' + shardLabel + '</span>' +
            '<span class="mono dl-progress-text">' + rightText + '</span>' +
            '</div>' +
            '<div class="dl-bar">' + (segmentsHtml || '<div class="dl-fill" style="width:' + pct + '%"></div>') + '</div>' +
            '</div>';
        }

        // Action button
        var actionHtml = '';
        if (m.status === 'loaded') {
          // already active — no button needed
        } else if (isReady) {
          actionHtml = '<button class="btn btn-sm btn-primary" data-select-model="' + escapeHtml(m.id) + '">Use</button>';
        } else if (isDownloading) {
          // Cancel download button
          actionHtml = '<button class="shard-cancel-btn" data-cancel-download="' + escapeHtml(m.id) + '" title="Cancel download">&times;</button>';
        } else if (m.source === 'network' || m.status === 'available' || m.status === 'partial') {
          actionHtml = '<button class="btn btn-sm" data-request-model="' + escapeHtml(m.id) + '">Download Missing</button>';
        }

        // Remove model button (for models with local shards, not currently downloading)
        var removeHtml = '';
        if (hostedShards > 0 && !isDownloading) {
          removeHtml = ' <button class="model-remove-btn" data-remove-model="' + escapeHtml(m.id) + '">Remove</button>';
        }

        var name = m.name || m.id;
        card.innerHTML =
          '<div class="model-header">' +
            '<span class="model-name">' + escapeHtml(name) + '</span>' +
            '<span>' + statusHtml + (actionHtml ? ' ' + actionHtml : '') + removeHtml + '</span>' +
          '</div>' +
          '<div class="model-meta">' + metaParts.map(function(p) { return '<span>' + p + '</span>'; }).join('') + fileIndicators + '</div>' +
          shardHtml + progressHtml;

        list.appendChild(card);
      });
    },

    /// Live-update shard cells and progress bars from WebSocket data without full re-render.
    updateShardsLive: function(acquisitions, shardRegistry, peerDownloads) {
      if (!acquisitions && !shardRegistry && !peerDownloads) return;

      // Update shard cells from acquisition progress (per-shard detail)
      if (acquisitions) {
        acquisitions.forEach(function(acq) {
          var modelId = acq.model_id;
          if (!modelId) return;
          var safeId = modelId.replace(/[^a-zA-Z0-9]/g, '_');

          // Update per-shard cells
          var shardDetails = acq.shard_details || [];
          var localCount = 0, peerCount = 0, dlCount = 0, peerDlCount = 0, queuedCount = 0, missingCount = 0;
          shardDetails.forEach(function(sd) {
            var cellId = safeId + '-' + sd.index;
            var cell = document.querySelector('[data-shard="' + cellId + '"]');
            if (!cell) return;

            var oldClass = cell.className.replace(/shard-cell\s*/, '').trim().split(/\s+/)[0] || 'missing';
            var newClass = 'missing';
            var label = '' + sd.index;
            var dlPct = sd.progress_pct || 0;

            if (sd.state === 'complete') { newClass = 'local'; localCount++; }
            else if (sd.state === 'downloading' || sd.state === 'verifying') {
              newClass = 'downloading'; dlCount++;
              label = dlPct + '%';
            } else if (sd.state === 'pending') {
              // Pending shards in an active acquisition are queued
              newClass = 'queued'; queuedCount++;
              label = '\u2022';
            } else if (sd.state === 'failed') { newClass = 'missing'; missingCount++; }
            else { missingCount++; }

            // Only update DOM if something changed
            if (oldClass !== newClass || cell.textContent !== label) {
              cell.className = 'shard-cell ' + newClass;
              cell.textContent = label;

              // Set gradient CSS variable for download progress
              if (newClass === 'downloading' || newClass === 'peer-downloading') {
                cell.style.setProperty('--dl-pct', dlPct + '%');
              } else {
                cell.style.removeProperty('--dl-pct');
              }

              // Update title
              var title = 'Shard ' + sd.index;
              if (newClass === 'local') title += ' \u2014 Complete';
              else if (newClass === 'downloading') title += ' \u2014 Downloading (' + dlPct + '%)';
              else if (newClass === 'queued') title += ' \u2014 Queued for download';
              else if (sd.state === 'failed') title += ' \u2014 Failed';
              else title += ' \u2014 Not available';
              cell.setAttribute('title', title);
            }
          });

          // Update progress bar — only allow forward progress to prevent jumping
          var progressEl = document.querySelector('[data-model-progress="' + safeId + '"]');
          if (progressEl && acq.total_bytes > 0) {
            var dlBytes = Math.min(acq.downloaded_bytes || 0, acq.total_bytes);
            var pct = Math.min(100, Math.round((dlBytes / acq.total_bytes) * 100));
            var lastPct = parseInt(progressEl.getAttribute('data-last-pct') || '0', 10);
            // Only update if progress moved forward (prevents jumping backward)
            if (pct >= lastPct) {
              progressEl.setAttribute('data-last-pct', '' + pct);
              var speed = acq.speed_bytes_per_sec || 0;
              var shardLabel = acq.downloaded_shards !== undefined ? ('Shard ' + acq.downloaded_shards + '/' + (acq.total_shards || shardDetails.length)) : 'Downloading';
              var etaStr = '';
              if (speed > 0 && acq.total_bytes > dlBytes) {
                var secsLeft = Math.round((acq.total_bytes - dlBytes) / speed);
                if (secsLeft >= 3600) etaStr = Math.floor(secsLeft / 3600) + 'h ' + Math.floor((secsLeft % 3600) / 60) + 'm';
                else if (secsLeft >= 60) etaStr = Math.floor(secsLeft / 60) + 'm ' + (secsLeft % 60) + 's';
                else etaStr = secsLeft + 's';
              }
              var textEl = progressEl.querySelector('.dl-progress-text');
              if (textEl) {
                var txt = formatBytes(dlBytes) + ' / ' + formatBytes(acq.total_bytes) + ' (' + pct + '%)';
                if (speed > 0) txt += ' \u00b7 ' + formatSpeed(speed);
                if (etaStr) txt += ' \u00b7 ETA ' + etaStr;
                textEl.textContent = txt;
              }
              var labelEl = progressEl.querySelector('.text-muted');
              if (labelEl) labelEl.textContent = shardLabel;
              // Update segmented bar fills
              var segs = progressEl.querySelectorAll('.dl-seg');
              if (segs.length > 0) {
                shardDetails.forEach(function(sd, i) {
                  if (segs[sd.index]) {
                    var segFill = segs[sd.index].querySelector('.dl-seg-fill');
                    var segPct = sd.local ? 100 : (sd.download ? (sd.download.progress_pct || 0) : 0);
                    if (segFill) segFill.style.width = segPct + '%';
                  }
                });
              } else {
                // Fallback: single fill bar
                var fillEl = progressEl.querySelector('.dl-fill');
                if (fillEl) fillEl.style.width = pct + '%';
              }
            }
          } else if (!progressEl && acq.total_bytes > 0 && acq.downloaded_bytes > 0) {
            // Insert progress bar if card exists but has no progress bar yet
            var card = document.querySelector('[data-model-id="' + modelId + '"]');
            if (card && !card.querySelector('.dl-progress')) {
              var dlBytes2 = Math.min(acq.downloaded_bytes, acq.total_bytes);
              var pct2 = Math.min(100, Math.round((dlBytes2 / acq.total_bytes) * 100));
              var speed2 = acq.speed_bytes_per_sec || 0;
              var shardLabel2 = acq.downloaded_shards !== undefined ? ('Shard ' + acq.downloaded_shards + '/' + (acq.total_shards || '?')) : 'Downloading';
              var progDiv = document.createElement('div');
              progDiv.className = 'dl-progress';
              progDiv.setAttribute('data-model-progress', safeId);
              progDiv.setAttribute('data-last-pct', '' + pct2);
              progDiv.innerHTML =
                '<div class="flex-between" style="font-size:0.75rem;margin-bottom:3px">' +
                '<span class="text-muted">' + shardLabel2 + '</span>' +
                '<span class="mono dl-progress-text">' + formatBytes(dlBytes2) + ' / ' + formatBytes(acq.total_bytes) + ' (' + pct2 + '%)' +
                (speed2 > 0 ? ' \u2014 ' + formatSpeed(speed2) : '') + '</span></div>' +
                '<div class="dl-bar"><div class="dl-fill" style="width:' + pct2 + '%"></div></div>';
              card.appendChild(progDiv);
              // Also add downloading class to card
              if (!card.classList.contains('downloading')) {
                card.classList.remove('partial');
                card.classList.add('downloading');
              }
            }
          }

          // Update legend if present
          var legendEl = document.querySelector('[data-model-legend="' + safeId + '"]');
          if (legendEl && shardDetails.length > 0) {
            var parts = [];
            if (localCount > 0) parts.push('<span class="leg-local">Local (' + localCount + ')</span>');
            if (peerCount > 0) parts.push('<span class="leg-peer">Peer (' + peerCount + ')</span>');
            if (dlCount > 0) parts.push('<span class="leg-dl">Downloading (' + dlCount + ')</span>');
            if (peerDlCount > 0) parts.push('<span class="leg-peer-dl">Peer DL (' + peerDlCount + ')</span>');
            if (queuedCount > 0) parts.push('<span class="leg-queued">Queued (' + queuedCount + ')</span>');
            if (missingCount > 0) parts.push('<span class="leg-missing">Missing (' + missingCount + ')</span>');
            legendEl.innerHTML = parts.join('');
          }
        });
      }

      // Update shard cells from shard registry changes (new shards from peers)
      if (shardRegistry) {
        Object.keys(shardRegistry).forEach(function(modelId) {
          var safeId = modelId.replace(/[^a-zA-Z0-9]/g, '_');
          var shards = shardRegistry[modelId] || [];
          shards.forEach(function(s) {
            var cellId = safeId + '-' + s.index;
            var cell = document.querySelector('[data-shard="' + cellId + '"]');
            if (!cell) return;

            // Only update to peer/local if not already downloading or local
            var current = cell.className;
            if (current.indexOf('downloading') >= 0 || current.indexOf('local') >= 0) return;

            if (s.local) {
              cell.className = 'shard-cell local';
              cell.textContent = '' + s.index;
              cell.setAttribute('title', 'Shard ' + s.index + ' — Stored locally');
            } else if (s.holders > 0 && current.indexOf('peer') < 0) {
              cell.className = 'shard-cell peer';
              cell.setAttribute('title', 'Shard ' + s.index + ' — Available from ' + s.holders + ' peer(s)');
            }
          });
        });
      }

      // Update shard cells with peer download progress (from gossip)
      if (peerDownloads && peerDownloads.length > 0) {
        peerDownloads.forEach(function(pd) {
          var safeId = pd.model_id.replace(/[^a-zA-Z0-9]/g, '_');
          var cellId = safeId + '-' + pd.shard_index;
          var cell = document.querySelector('[data-shard="' + cellId + '"]');
          if (!cell) return;

          // Don't overwrite local or our-own-download state
          var current = cell.className;
          if (current.indexOf('local') >= 0 || current.indexOf(' downloading') >= 0) return;

          var pdPct = pd.progress_pct || 0;
          cell.className = 'shard-cell peer-downloading';
          cell.style.setProperty('--dl-pct', pdPct + '%');
          cell.textContent = pdPct + '%';
          cell.setAttribute('title', 'Shard ' + pd.shard_index + ' \u2014 Peer ' + pd.node_id.substring(0, 8) + ' downloading (' + pdPct + '%)');
        });
      }
    },


    loadNetworkData: async function() {
      try {
        var resp = await fetch('/api/admin/peers');
        var peers = await resp.json();
        var list = document.getElementById('peers-list');
        if (peers && peers.length > 0) {
          list.innerHTML = '';
          peers.forEach(function(p) {
            var div = document.createElement('div');
            div.style.cssText = 'margin-bottom:10px;padding:8px 10px;background:var(--bg-tertiary);border-radius:var(--radius);border:1px solid var(--border)';
            var statusDot = '<span class="status-dot ' + (p.healthy ? 'online' : 'degraded') + '"></span>';
            var nodeId = '<span class="mono" style="font-size:0.8rem">' + escapeHtml(p.node_id || 'unknown') + '</span>';
            var details = '';
            if (p.gpu) details += '<div style="font-size:0.75rem;color:var(--text-secondary);margin-top:3px">GPU: ' + escapeHtml(p.gpu) + '</div>';
            div.innerHTML = statusDot + nodeId + details;
            list.appendChild(div);
          });
        }
      } catch (e) {}

    },

    updateAcquisitionProgress: function(acquisitions) {
      if (!acquisitions || acquisitions.length === 0) return;
      acquisitions.forEach(function(status) {
        var modelId = status.model_id;
        if (!modelId) return;
        if (!activeAcquisitions[modelId]) {
          if (status.state === 'complete') return;
          activeAcquisitions[modelId] = { started: Date.now() };
        }
        dashboard.renderAcquisitionPanel(modelId, status);
        if (status.state === 'complete') {
          setTimeout(function() { delete activeAcquisitions[modelId]; dashboard.loadInitial(); }, 3000);
        } else if (status.state && status.state.failed) {
          setTimeout(function() { delete activeAcquisitions[modelId]; }, 10000);
        }
      });
    },

    renderAcquisitionPanel: function(modelId, status) {
      // Progress is now shown inline in model cards only — no separate banner panels.
      // This method updates the model card's progress bar and status badge in-place.
      if (!status) return;
      var safeId = modelId.replace(/[^a-zA-Z0-9]/g, '_');
      var card = document.querySelector('[data-model-id="' + modelId + '"]');
      if (!card) return; // card not rendered yet — will show on next loadInitial

      var stateName = typeof status.state === 'string' ? status.state : (status.state && status.state.failed ? 'failed' : 'unknown');

      // If complete, refresh model list to pick up new shard state
      if (stateName === 'complete') {
        if (!card.classList.contains('ready')) {
          setTimeout(function() { dashboard.loadInitial(); }, 1500);
        }
        return;
      }

      // Ensure card has downloading class
      if (!card.classList.contains('downloading')) {
        card.classList.remove('partial');
        card.classList.add('downloading');
      }

      // Update or insert progress bar in card
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

      var speedStr = speed > 0 ? ' - ' + formatSpeed(speed) : '';
      progressEl.innerHTML =
        '<div class="flex-between" style="font-size:0.75rem;margin-bottom:3px">' +
        '<span class="text-muted">Downloading shard</span>' +
        '<span class="mono dl-progress-text">' + formatBytes(dlBytes) + ' / ' + formatBytes(totalBytes) + ' (' + pct + '%)' + speedStr + '</span>' +
        '</div>' +
        '<div class="dl-bar"><div class="dl-fill" style="width:' + pct + '%"></div></div>';

      // Remove any stale banner panels (legacy cleanup)
      var oldPanel = document.getElementById('acq-panel-' + safeId);
      if (oldPanel) oldPanel.remove();
    }
  };

  // ========================================================================
  // HuggingFace Module — model search and download
  // ========================================================================
  var hf = {
    search: async function() {
      var query = document.getElementById('hf-search-input').value.trim();
      if (!query) return;

      var results = document.getElementById('hf-results');
      var loading = document.getElementById('hf-loading');
      results.innerHTML = '';
      loading.classList.remove('hidden');

      try {
        var resp = await fetch('/api/admin/hf/search?q=' + encodeURIComponent(query));
        loading.classList.add('hidden');

        if (!resp.ok) {
          var errBody = await resp.text();
          try { var errJson = JSON.parse(errBody); errBody = errJson.error ? errJson.error.message : errBody; } catch (e2) {}
          results.innerHTML = '<div class="empty-state"><p>Search failed: ' + escapeHtml(errBody) + '</p></div>';
          return;
        }

        var data = await resp.json();

        if (!Array.isArray(data) || data.length === 0) {
          results.innerHTML = '<div class="empty-state"><p>No GGUF models found for "' + escapeHtml(query) + '"</p></div>';
          return;
        }

        // Fetch pool VRAM info for fitness display
        var poolVram = 0;
        try {
          var storageResp = await fetch('/api/admin/shard-storage');
          if (storageResp.ok) {
            var storageData = await storageResp.json();
            poolVram = storageData.pool_vram_mb || 0;
          }
        } catch (e2) {}

        results.innerHTML = '';
        data.forEach(function(model) {
          var card = document.createElement('div');
          card.className = 'hf-model-card';
          var sizeStr = model.size_bytes ? formatBytes(model.size_bytes) : 'Unknown size';
          var downloads = model.downloads ? model.downloads.toLocaleString() + ' downloads' : '';

          // Estimate VRAM requirement (model size * 1.15 overhead)
          var vramTag = '';
          if (model.size_bytes && model.size_bytes > 0) {
            var estVramMb = Math.ceil(model.size_bytes * 1.15 / (1024 * 1024));
            var estStr = escapeHtml(formatMB(estVramMb));
            var poolStr = escapeHtml(formatMB(poolVram));
            if (poolVram > 0) {
              if (estVramMb <= poolVram) {
                vramTag = '<span style="color:var(--green)" title="Fits in network VRAM pool (' + poolStr + ')">' + estStr + ' VRAM</span>';
              } else {
                vramTag = '<span style="color:var(--red)" title="Exceeds network VRAM pool (' + poolStr + '). Can still download but won\'t run yet.">' + estStr + ' VRAM (exceeds pool)</span>';
              }
            } else {
              vramTag = '<span class="text-muted">' + estStr + ' VRAM est.</span>';
            }
          }

          card.innerHTML = '<div class="hf-model-info">' +
            '<div class="hf-model-name">' + escapeHtml(model.repo_id || model.id) + '</div>' +
            '<div class="hf-model-meta">' +
            (model.filename ? '<span class="mono">' + escapeHtml(model.filename) + '</span>' : '') +
            '<span>' + sizeStr + '</span>' +
            (downloads ? '<span>' + downloads + '</span>' : '') +
            (vramTag ? '<span>' + vramTag + '</span>' : '') +
            '</div>' +
            '</div>' +
            '<div class="hf-model-actions">' +
            '<select class="hf-download-mode" id="dl-mode-' + escapeHtml(model.repo_id || model.id).replace(/[^a-zA-Z0-9]/g, '_') + '">' +
            '<option value="shards">Download shards (rarest first)</option>' +
            '<option value="full">Download full model</option>' +
            '</select>' +
            '<button class="btn btn-sm btn-primary" data-hf-download="' + escapeHtml(model.repo_id || model.id) + '" data-hf-filename="' + escapeHtml(model.filename || '') + '">Download</button>' +
            '</div>';
          results.appendChild(card);
        });
      } catch (e) {
        loading.classList.add('hidden');
        results.innerHTML = '<div class="empty-state"><p>Search failed: ' + escapeHtml(e.message) + '</p></div>';
      }
    },

    download: async function(repoId, filename) {
      try {
        var safeKey = (repoId || '').replace(/[^a-zA-Z0-9]/g, '_');
        var modeEl = document.getElementById('dl-mode-' + safeKey);
        var mode = modeEl ? modeEl.value : 'shards';

        if (mode === 'full') {
          // Full model download (single GGUF file) — user explicitly chose this
          var resp = await authFetch('/api/admin/hf/download', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ repo_id: repoId, filename: filename }),
          });
          var data = await resp.json();
          if (data.status === 'started' || data.status === 'acquiring') {
            ui.showBanner('success', 'Full model download started');
            ui.closeModelBrowser();
          } else {
            ui.showBanner('warning', data.message || 'Download could not be started');
          }
        } else {
          // Default: shard-based download. Probe first to discover shard count,
          // then download all shards. Other nodes auto-acquire via gossip.
          ui.showBanner('info', 'Probing model...');
          var probeResp = await fetch('/api/admin/hf/probe?repo_id=' + encodeURIComponent(repoId) + '&filename=' + encodeURIComponent(filename));
          var probeData = await probeResp.json();
          if (probeData.status !== 'ok' || !probeData.shard_count) {
            ui.showBanner('error', probeData.message || 'Failed to probe model');
            return;
          }
          // Build array of all shard indices [0, 1, 2, ...]
          var shardIndices = [];
          for (var i = 0; i < probeData.shard_count; i++) shardIndices.push(i);

          var resp = await authFetch('/api/admin/hf/download-shards', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ repo_id: repoId, filename: filename, shards: shardIndices }),
          });
          var data = await resp.json();
          if (data.status === 'started') {
            ui.showBanner('success', 'Downloading ' + probeData.shard_count + ' shards from HuggingFace');
            ui.closeModelBrowser();
          } else {
            ui.showBanner('warning', data.message || 'Download could not be started');
          }
        }
      } catch (e) {
        ui.showBanner('error', 'Download failed: ' + e.message);
      }
    }
  };

  // ========================================================================
  // Settings Module
  // ========================================================================
  var settings = {
    init: function() {
      // Toggle storage limit field visibility based on auto-manage setting
      var autoSelect = document.getElementById('settings-auto-shards');
      if (autoSelect) {
        autoSelect.addEventListener('change', function() {
          var isOn = this.value === 'on';
          document.getElementById('settings-auto-manage-storage-group').style.display = isOn ? '' : 'none';
          document.getElementById('settings-storage-info').classList.toggle('hidden', !isOn);
          if (isOn) settings.loadStorageInfo();
        });
      }
    },

    load: async function() {
      try {
        var resp = await fetch('/api/admin/config');
        var data = await resp.json();
        document.getElementById('settings-contribution').value = data.contribution || 'moderate';
        document.getElementById('settings-max-requests').value = data.max_concurrent_requests || 10;
        document.getElementById('settings-bandwidth').value = data.max_bandwidth_mbps || 0;
        document.getElementById('settings-disk').value = data.max_disk_mb || 50000;
        var autoManage = data.auto_manage_shards ? 'on' : 'off';
        document.getElementById('settings-auto-shards').value = autoManage;
        document.getElementById('settings-auto-manage-storage').value = data.auto_manage_max_storage_mb || 0;
        // Show/hide storage group
        var isOn = autoManage === 'on';
        document.getElementById('settings-auto-manage-storage-group').style.display = isOn ? '' : 'none';
        document.getElementById('settings-storage-info').classList.toggle('hidden', !isOn);
        if (isOn) settings.loadStorageInfo();
      } catch (e) {}
      // Load API key
      settings.loadApiKey();
    },

    _apiKeyFull: '',

    loadApiKey: async function() {
      var keyEl = document.getElementById('settings-api-key');
      if (!keyEl) return;
      try {
        var resp = await fetch('/api/admin/api-key');
        if (resp.ok) {
          var data = await resp.json();
          var key = data.api_key || '';
          settings._apiKeyFull = key;
          keyEl.value = key ? key.substring(0, 4) + '****' + key.substring(key.length - 4) : 'No API key';
        } else {
          keyEl.value = 'Unavailable';
        }
      } catch (e) {
        keyEl.value = 'Error loading';
      }
    },

    copyApiKey: async function() {
      var btn = document.getElementById('btn-copy-api-key');
      if (!settings._apiKeyFull) return;
      try {
        await navigator.clipboard.writeText(settings._apiKeyFull);
        if (btn) {
          btn.textContent = 'Copied!';
          btn.style.color = 'var(--green)';
          btn.style.borderColor = 'var(--green)';
          setTimeout(function() {
            btn.textContent = 'Copy';
            btn.style.color = '';
            btn.style.borderColor = '';
          }, 2000);
        }
      } catch (e) {
        if (btn) btn.textContent = 'Failed';
        setTimeout(function() { if (btn) btn.textContent = 'Copy'; }, 2000);
      }
    },

    loadStorageInfo: async function() {
      try {
        var resp = await fetch('/api/admin/shard-storage');
        var data = await resp.json();
        document.getElementById('settings-storage-used').textContent = formatBytes(data.disk_usage_bytes || 0);
        var maxMb = data.auto_manage_max_storage_mb || 0;
        document.getElementById('settings-storage-max').textContent = maxMb > 0 ? formatMB(maxMb) : '50% of disk limit';

        // Show pool VRAM capacity
        var poolVram = data.pool_vram_mb || 0;
        var localVram = data.local_vram_mb || 0;
        var peerCount = data.peer_count || 0;
        var poolEl = document.getElementById('settings-pool-vram');
        if (poolEl) {
          if (poolVram > 0) {
            poolEl.innerHTML = '<strong>' + formatMB(poolVram) + '</strong> total VRAM' +
              ' (local: ' + formatMB(localVram) + ', ' + peerCount + ' peer' + (peerCount !== 1 ? 's' : '') + ')';
          } else {
            poolEl.innerHTML = '<span class="text-muted">No GPU detected</span>';
          }
        }

        var modelsDiv = document.getElementById('settings-storage-models');
        modelsDiv.innerHTML = '';
        if (data.models && data.models.length > 0) {
          data.models.forEach(function(m) {
            if (m.local_shards > 0) {
              var vramNeeded = m.estimated_vram_mb || 0;
              var fits = poolVram > 0 && vramNeeded <= poolVram;
              var tooLarge = poolVram > 0 && vramNeeded > poolVram;
              var vramTag = '';
              if (vramNeeded > 0) {
                var vramStr = escapeHtml(formatMB(vramNeeded));
                var poolVramStr = escapeHtml(formatMB(poolVram));
                if (fits) {
                  vramTag = ' <span style="color:var(--green)">' + vramStr + ' VRAM</span>';
                } else if (tooLarge) {
                  vramTag = ' <span style="color:var(--red)" title="Exceeds pool VRAM (' + poolVramStr + ')">' + vramStr + ' VRAM</span>';
                } else {
                  vramTag = ' <span class="text-muted">' + vramStr + ' VRAM</span>';
                }
              }
              var div = document.createElement('div');
              div.className = 'flex-between';
              div.style.cssText = 'padding:2px 0';
              div.innerHTML = '<span>' + escapeHtml(m.name || m.id) + '</span>' +
                '<span class="text-muted">' + m.local_shards + '/' + m.shard_count + ' shards &middot; ' + formatBytes(m.local_bytes) + vramTag + '</span>';
              modelsDiv.appendChild(div);
            }
          });
          if (modelsDiv.children.length === 0) {
            modelsDiv.innerHTML = '<span class="text-muted">No local shards yet</span>';
          }
        } else {
          modelsDiv.innerHTML = '<span class="text-muted">No models registered</span>';
        }
      } catch (e) {}
    },

    save: async function() {
      var autoManageOn = document.getElementById('settings-auto-shards').value === 'on';
      var config = {
        contribution: document.getElementById('settings-contribution').value,
        max_concurrent_requests: parseInt(document.getElementById('settings-max-requests').value, 10),
        max_bandwidth_mbps: parseInt(document.getElementById('settings-bandwidth').value, 10),
        max_disk_mb: parseInt(document.getElementById('settings-disk').value, 10),
        auto_manage_shards: autoManageOn,
        auto_manage_max_storage_mb: autoManageOn ? parseInt(document.getElementById('settings-auto-manage-storage').value, 10) || 0 : 0,
      };

      try {
        var resp = await authFetch('/api/admin/config', {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(config),
        });
        if (resp.ok) {
          ui.showBanner('success', 'Settings saved');
          ui.closeSettings();
        } else {
          ui.showBanner('error', 'Failed to save settings');
        }
      } catch (e) {
        ui.showBanner('error', 'Error: ' + e.message);
      }

      // Save nickname if provided
      await identity.saveNickname();
    }
  };

  // ========================================================================
  // Setup Wizard Module
  // ========================================================================
  var setup = {
    currentStep: 1,
    hwData: null,

    init: function() {
      if (localStorage.getItem(SETUP_DONE_KEY) === 'true') return;
      document.getElementById('setup-modal').classList.remove('hidden');
      setup.detectHardware();

      document.getElementById('contribution-slider').addEventListener('input', function() {
        var levels = ['Minimal', 'Moderate', 'Maximum'];
        var descs = [
          'Low impact: uses minimal resources. Best for shared or low-spec machines.',
          'Balanced: uses ~50% of available resources. Good for most users.',
          'Full power: uses all available resources. Best for dedicated nodes.',
        ];
        var val = parseInt(this.value, 10);
        document.getElementById('contribution-label').textContent = levels[val];
        document.getElementById('contribution-desc').textContent = descs[val];
      });
    },

    detectHardware: async function() {
      try {
        var resp = await fetch('/api/admin/stats');
        var data = await resp.json();
        setup.hwData = data.hardware || {};
        document.getElementById('hw-gpu').textContent = setup.hwData.gpu_name || 'No GPU (CPU mode)';
        document.getElementById('hw-vram').textContent = setup.hwData.gpu_vram_mb ? setup.hwData.gpu_vram_mb + ' MB' : 'N/A';
        document.getElementById('hw-ram').textContent = formatMB(setup.hwData.total_ram_mb || 0);
        document.getElementById('hw-disk').textContent = formatMB(setup.hwData.available_disk_mb || 0);
      } catch (e) {
        document.getElementById('hw-gpu').textContent = 'Detection failed';
        setup.hwData = {};
      }
      document.getElementById('hw-loading').classList.add('hidden');
      document.getElementById('hw-results').classList.remove('hidden');
    },

    nextStep: function() {
      if (setup.currentStep === 4) {
        setup.submit();
        return;
      }
      setup.currentStep++;
      setup.updateUI();
      if (setup.currentStep === 3) setup.loadModelSelection();
      if (setup.currentStep === 4) setup.populateSummary();
    },

    prevStep: function() {
      if (setup.currentStep > 1) {
        setup.currentStep--;
        setup.updateUI();
      }
    },

    updateUI: function() {
      for (var i = 1; i <= 4; i++) {
        var body = document.getElementById('step-' + i);
        var indicator = document.querySelector('[data-step="' + i + '"]');
        if (i === setup.currentStep) {
          body.classList.remove('hidden');
          indicator.classList.add('active');
          indicator.classList.remove('done');
        } else if (i < setup.currentStep) {
          body.classList.add('hidden');
          indicator.classList.remove('active');
          indicator.classList.add('done');
        } else {
          body.classList.add('hidden');
          indicator.classList.remove('active', 'done');
        }
      }
      var connectors = document.querySelectorAll('.wizard-connector');
      connectors.forEach(function(c, idx) { c.classList.toggle('done', idx + 1 < setup.currentStep); });
      document.getElementById('btn-prev').classList.toggle('hidden', setup.currentStep === 1);
      document.getElementById('btn-next').textContent = setup.currentStep === 4 ? 'Start SwarmLLM' : 'Continue';
    },

    loadModelSelection: async function() {
      var list = document.getElementById('setup-model-list');
      list.innerHTML = '<p class="text-muted">Loading available models...</p>';
      try {
        var resp = await fetch('/api/admin/models');
        var models = await resp.json();
        if (!models || models.length === 0) {
          list.innerHTML = '<div class="empty-state" style="padding:20px 0">' +
            '<p style="margin-bottom:8px">No models available yet.</p>' +
            '<p class="text-muted" style="font-size:0.85rem">You can download models from HuggingFace after setup using the <strong>Browse Models</strong> button on the dashboard.</p>' +
            '</div>';
        } else {
          list.innerHTML = '';
          models.forEach(function(m) {
            var div = document.createElement('div');
            div.style.cssText = 'padding:8px 10px;margin-bottom:6px;background:var(--bg-tertiary);border-radius:var(--radius);border:1px solid var(--border)';
            var name = m.name || m.id;
            if (name.length > 50) name = name.substring(0, 50) + '...';
            var size = m.total_size_bytes ? formatBytes(m.total_size_bytes) : '';
            var status = m.status === 'loaded' ? '<span class="text-green" style="font-size:0.8rem">Loaded</span>' : '<span class="text-muted" style="font-size:0.8rem">' + (m.status || 'available') + '</span>';
            div.innerHTML = '<div style="display:flex;justify-content:space-between;align-items:center"><strong style="font-size:0.9rem">' + escapeHtml(name) + '</strong>' + status + '</div>' +
              (size ? '<div class="text-muted" style="font-size:0.8rem">' + size + '</div>' : '');
            list.appendChild(div);
          });
        }
      } catch (e) {
        list.innerHTML = '<div class="empty-state"><p>Could not load models. You can browse models after setup.</p></div>';
      }
    },

    populateSummary: function() {
      var levels = ['minimal', 'moderate', 'maximum'];
      var val = parseInt(document.getElementById('contribution-slider').value, 10);
      document.getElementById('summary-contribution').textContent = capitalize(levels[val]);
      document.getElementById('summary-gpu').textContent = setup.hwData && setup.hwData.gpu_name ? setup.hwData.gpu_name : 'CPU only';
      document.getElementById('summary-ram').textContent = formatMB(setup.hwData ? setup.hwData.total_ram_mb || 0 : 0);
      document.getElementById('summary-disk').textContent = formatMB(setup.hwData ? setup.hwData.available_disk_mb || 0 : 0);
      var autoManage = document.getElementById('setup-auto-manage').checked;
      document.getElementById('summary-auto-manage').textContent = autoManage ? 'Enabled' : 'Disabled';
      document.getElementById('summary-models').textContent = 'Default configuration';
    },

    submit: async function() {
      var levels = ['minimal', 'moderate', 'maximum'];
      var level = levels[parseInt(document.getElementById('contribution-slider').value, 10)];
      var autoManage = document.getElementById('setup-auto-manage').checked;
      try {
        await authFetch('/api/admin/config', {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            contribution: level,
            auto_manage_shards: autoManage,
          }),
        });
      } catch (e) {}
      localStorage.setItem(SETUP_DONE_KEY, 'true');
      // Also persist to server so other clients / restarts see setup as done
      try {
        await authFetch('/api/admin/config', {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ setup_done: true }),
        });
      } catch (e) {}
      document.getElementById('setup-modal').classList.add('hidden');
    }
  };

  // ========================================================================
  // WebSocket — real-time updates
  // ========================================================================
  var wsWasConnected = false;
  var wsBannerTimer = null;

  function showWsBanner(type, text) {
    var banner = document.getElementById('ws-banner');
    if (!banner) return;
    if (wsBannerTimer) { clearTimeout(wsBannerTimer); wsBannerTimer = null; }
    banner.innerHTML = '<div class="ws-banner-' + escapeHtml(type) + '">' + escapeHtml(text) + '</div>';
    banner.classList.add('show');
  }

  function hideWsBanner(delay) {
    var banner = document.getElementById('ws-banner');
    if (!banner) return;
    if (wsBannerTimer) clearTimeout(wsBannerTimer);
    wsBannerTimer = setTimeout(function() {
      banner.classList.remove('show');
    }, delay || 0);
  }

  function connectWebSocket() {
    var protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    ws = new WebSocket(protocol + '//' + window.location.host + '/api/admin/ws');

    ws.onopen = function() {
      wsHealthy = true;
      // Pause REST polling while WebSocket is delivering live updates
      pollTimers.forEach(function(t) { clearInterval(t); });
      pollTimers = [];
      if (wsWasConnected) {
        showWsBanner('connected', 'Connected');
        hideWsBanner(2000);
      }
      wsWasConnected = true;
    };

    ws.onmessage = function(event) {
      try {
        var msg = JSON.parse(event.data);
        if (msg.type === 'stats_update') {
          dashboard.updateStats(msg.data);
          if (msg.data.acquisitions) dashboard.updateAcquisitionProgress(msg.data.acquisitions);
          // Live-update shard grid cells and progress bars without full re-render
          dashboard.updateShardsLive(msg.data.acquisitions, msg.data.shard_registry || null, msg.data.peer_downloads || null);
          if (msg.data.region_summary && activeTab === 'network-map') {
            networkMap.updateFromWs(msg.data.region_summary);
          }
        }
      } catch (e) {}
    };

    ws.onclose = function() {
      wsHealthy = false;
      if (wsWasConnected) {
        showWsBanner('disconnected', 'Connection lost \u2014 reconnecting...');
      }
      // Resume REST polling as fallback while WebSocket is disconnected
      startPolling();
      setTimeout(connectWebSocket, 3000);
    };
    ws.onerror = function() { ws.close(); };
  }

  function startPolling() {
    if (pollTimers.length > 0) return; // already polling
    pollTimers.push(setInterval(dashboard.loadInitial, 30000));
    pollTimers.push(setInterval(loadModels, 30000));
  }

  // ========================================================================
  // Model loading + selection
  // ========================================================================
  async function loadModels() {
    try {
      // Fetch admin model list to check readiness status
      var adminResp = await fetch('/api/admin/models');
      var adminModels = adminResp.ok ? await adminResp.json() : [];

      // Build set of ready model IDs (status: loaded, ready, or all shards available)
      var readySet = {};
      adminModels.forEach(function(m) {
        var isReady = m.status === 'loaded' || m.status === 'ready' ||
          (m.global_available === m.shard_count && m.shard_count > 0);
        if (isReady) readySet[m.id] = true;
      });

      // Build model selector from admin model list (auth-exempt)
      var sel = document.getElementById('model-select');
      sel.innerHTML = '';

      var readyModels = adminModels.filter(function(m) { return readySet[m.id]; });

      if (readyModels.length > 0) {
        var savedModel = null;
        try { savedModel = localStorage.getItem('swarmllm_current_model'); } catch (e) {}
        var found = savedModel && readyModels.some(function(m) { return m.id === savedModel; });
        currentModel = found ? savedModel : readyModels[0].id;
        readyModels.forEach(function(m) {
          var opt = document.createElement('option');
          opt.value = m.id;
          opt.textContent = m.id.length > 30 ? m.id.substring(0, 30) + '...' : m.id;
          sel.appendChild(opt);
        });
        sel.value = currentModel;
      } else if (adminModels.length > 0) {
        sel.innerHTML = '<option value="" disabled>No models ready</option>';
      } else {
        sel.innerHTML = '<option value="">No model loaded</option>';
      }
    } catch (e) {}
  }

  async function requestModel(modelId) {
    try {
      var resp = await authFetch('/api/admin/models/' + encodeURIComponent(modelId) + '/add', { method: 'POST' });
      var data = await resp.json();
      if (data.status === 'acquiring') {
        activeAcquisitions[modelId] = { started: Date.now() };
        dashboard.renderAcquisitionPanel(modelId, null);
      } else {
        ui.showBanner('warning', data.message || 'Model acquisition unavailable');
      }
    } catch (e) {
      ui.showBanner('error', 'Failed to request model: ' + e.message);
    }
  }

  function selectModel(modelId) {
    currentModel = modelId;
    try { localStorage.setItem('swarmllm_current_model', modelId); } catch (e) {}
    var sel = document.getElementById('model-select');
    if (sel) {
      // Ensure model is in the dropdown
      var found = false;
      for (var i = 0; i < sel.options.length; i++) {
        if (sel.options[i].value === modelId) { found = true; break; }
      }
      if (!found) {
        var opt = document.createElement('option');
        opt.value = modelId;
        opt.textContent = modelId.length > 30 ? modelId.substring(0, 30) + '...' : modelId;
        sel.appendChild(opt);
      }
      sel.value = modelId;
    }
    ui.showBanner('success', 'Model selected: ' + modelId);
    // Refresh model list from server
    loadModels();
  }

  // ========================================================================
  // Cancel Download
  // ========================================================================
  async function cancelDownload(modelId) {
    if (!confirm('Cancel download for ' + modelId + '?')) return;
    try {
      // NOTE: Backend endpoint POST /api/admin/downloads/{model_id}/cancel
      // does not exist yet — backend work needed to implement cancellation.
      var resp = await authFetch('/api/admin/downloads/' + encodeURIComponent(modelId) + '/cancel', { method: 'POST' });
      if (resp.ok) {
        ui.showBanner('success', 'Download cancelled');
        // Remove progress UI from the card
        var safeId = modelId.replace(/[^a-zA-Z0-9]/g, '_');
        var card = document.querySelector('[data-model-id="' + modelId + '"]');
        if (card) {
          var progress = card.querySelector('.dl-progress');
          if (progress) progress.remove();
          card.classList.remove('downloading');
        }
        delete activeAcquisitions[modelId];
        setTimeout(function() { dashboard.loadInitial(); }, 1000);
      } else {
        var errData = await resp.json().catch(function() { return {}; });
        ui.showBanner('error', errData.error ? errData.error.message : 'Failed to cancel download');
      }
    } catch (e) {
      ui.showBanner('error', 'Cancel failed: ' + e.message);
    }
  }

  // ========================================================================
  // Remove Model
  // ========================================================================
  async function removeModel(modelId) {
    if (!confirm('Remove all local shards for ' + modelId + '? This cannot be undone.')) return;
    try {
      // NOTE: Backend endpoint DELETE /api/admin/models/{model_id}
      // does not exist yet — backend work needed to implement model removal.
      var resp = await authFetch('/api/admin/models/' + encodeURIComponent(modelId), { method: 'DELETE' });
      if (resp.ok) {
        ui.showBanner('success', 'Model removed: ' + modelId);
        // Remove the card from UI
        var card = document.querySelector('[data-model-id="' + modelId + '"]');
        if (card) card.remove();
        setTimeout(function() { dashboard.loadInitial(); }, 1000);
      } else {
        var errData = await resp.json().catch(function() { return {}; });
        ui.showBanner('error', errData.error ? errData.error.message : 'Failed to remove model');
      }
    } catch (e) {
      ui.showBanner('error', 'Remove failed: ' + e.message);
    }
  }

  // ========================================================================
  // Shutdown
  // ========================================================================
  async function shutdown() {
    if (!confirm('Shut down SwarmLLM node?')) return;
    try {
      await authFetch('/api/admin/shutdown', { method: 'POST' });
      document.body.innerHTML = '<div style="display:flex;align-items:center;justify-content:center;height:100vh;color:var(--text-muted);font-size:1.2rem">SwarmLLM has been shut down.</div>';
    } catch (e) {
      ui.showBanner('error', 'Shutdown failed: ' + e.message);
    }
  }

  // ========================================================================
  // Helpers
  // ========================================================================
  // escapeHtml is defined once at the top of the IIFE (handles null via || '')

  function setTierBadge(elementId, tier) {
    var el = document.getElementById(elementId);
    if (!el) return;
    el.textContent = capitalize(tier);
    el.className = 'tier-badge ' + tier.toLowerCase();
  }

  function formatUptime(seconds) {
    if (seconds < 60) return seconds + 's';
    if (seconds < 3600) return Math.floor(seconds / 60) + 'm';
    var h = Math.floor(seconds / 3600);
    var m = Math.floor((seconds % 3600) / 60);
    if (h >= 24) { return Math.floor(h / 24) + 'd ' + (h % 24) + 'h'; }
    return h + 'h ' + m + 'm';
  }

  function formatMB(mb) {
    if (!mb || mb === 0) return '\u2014';
    if (mb >= 1024) return (mb / 1024).toFixed(1) + ' GB';
    return mb + ' MB';
  }

  function formatBytes(bytes) {
    if (!bytes || bytes === 0) return '\u2014';
    if (bytes >= 1073741824) return (bytes / 1073741824).toFixed(1) + ' GB';
    if (bytes >= 1048576) return (bytes / 1048576).toFixed(1) + ' MB';
    return Math.round(bytes / 1024) + ' KB';
  }

  function formatSpeed(bytesPerSec) {
    if (bytesPerSec >= 1048576) return (bytesPerSec / 1048576).toFixed(1) + ' MB/s';
    if (bytesPerSec >= 1024) return Math.round(bytesPerSec / 1024) + ' KB/s';
    return bytesPerSec + ' B/s';
  }

  function capitalize(s) { return s.charAt(0).toUpperCase() + s.slice(1); }

  function renderSparkline(containerId, data) {
    var container = document.getElementById(containerId);
    if (!data || data.length === 0) return;
    var max = Math.max.apply(null, data) || 1;
    container.innerHTML = '';
    data.forEach(function(val) {
      var bar = document.createElement('div');
      bar.className = 'bar';
      bar.style.height = Math.max(2, (val / max) * 36) + 'px';
      container.appendChild(bar);
    });
  }

  function appendMessageToDOM(role, content) {
    var container = document.getElementById('chat-messages');
    var empty = document.getElementById('chat-empty');
    if (empty) empty.style.display = 'none';

    var div = document.createElement('div');
    div.className = 'chat-msg ' + role;
    var label = role === 'user' ? 'You' : 'Assistant';
    div.innerHTML = '<div class="msg-role">' + label + '</div><div class="msg-content"></div>';
    div.querySelector('.msg-content').textContent = content;
    container.appendChild(div);
    chat.scrollToBottom();
    return div;
  }

  function createEmptyState() {
    var div = document.createElement('div');
    div.className = 'chat-empty';
    div.id = 'chat-empty';
    div.innerHTML = '<div class="chat-empty-icon">&#11088;</div>' +
      '<div style="font-size:1.2rem;font-weight:600;color:var(--text-primary)">SwarmLLM Chat</div>' +
      '<div>Send a message to start a conversation</div>';
    return div;
  }

  var inputEl = null;
  function autoResizeInput() {
    if (!inputEl) inputEl = document.getElementById('chat-input');
    if (!inputEl) return;
    inputEl.style.height = 'auto';
    inputEl.style.height = Math.min(inputEl.scrollHeight, 200) + 'px';
  }

  function updateTokenCounter() {
    var el = document.getElementById('token-counter');
    if (!el) return;
    var input = document.getElementById('chat-input');
    if (!input) return;
    var text = input.value;
    if (!text) { el.textContent = ''; el.className = 'token-counter'; return; }
    var tokens = Math.ceil(text.length / 4);
    el.textContent = '~' + tokens + ' tokens';
    if (tokens > 7000) { el.className = 'token-counter danger'; }
    else if (tokens > 3000) { el.className = 'token-counter warn'; }
    else { el.className = 'token-counter'; }
  }

  // ========================================================================
  // Identity Module — nickname, leaderboard
  // ========================================================================
  var identity = {
    loadNickname: async function() {
      try {
        var resp = await fetch('/api/identity/nickname');
        if (!resp.ok) return;
        var data = await resp.json();
        var nickEl = document.getElementById('settings-nickname');
        var visEl = document.getElementById('settings-visibility');
        if (nickEl && data.nickname) nickEl.value = data.nickname;
        if (visEl && data.visibility) visEl.value = data.visibility;
      } catch (e) {
        // ignore
      }
    },

    saveNickname: async function() {
      var nickEl = document.getElementById('settings-nickname');
      var visEl = document.getElementById('settings-visibility');
      if (!nickEl) return;
      var nickname = nickEl.value.trim();

      if (!nickname) {
        // Delete nickname (go anonymous)
        try {
          await authFetch('/api/identity/nickname', { method: 'DELETE' });
        } catch (e) { /* ignore */ }
        return;
      }

      try {
        var resp = await authFetch('/api/identity/nickname', {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            nickname: nickname,
            visibility: visEl ? visEl.value : 'nickname',
          }),
        });
        if (!resp.ok) {
          var err = await resp.json().catch(function() { return {}; });
          ui.showBanner('error', err.error ? err.error.message : 'Failed to set nickname');
        }
      } catch (e) {
        ui.showBanner('error', 'Error saving nickname: ' + e.message);
      }
    },

    loadLeaderboard: async function() {
      var tbody = document.getElementById('leaderboard-body');
      if (!tbody) return;

      try {
        var resp = await fetch('/api/identity/leaderboard?limit=50');
        if (!resp.ok) { tbody.innerHTML = '<tr><td colspan="4" class="text-muted" style="text-align:center">Failed to load</td></tr>'; return; }
        var data = await resp.json();
        var entries = data.leaderboard || [];

        if (entries.length === 0) {
          tbody.innerHTML = '<tr><td colspan="4" class="text-muted" style="text-align:center;padding:24px">No data yet</td></tr>';
          return;
        }

        var html = '';
        for (var i = 0; i < entries.length; i++) {
          var e = entries[i];
          var tierClass = (e.tier || 'silver').toLowerCase();
          html += '<tr>'
            + '<td class="mono">' + (e.rank || i+1) + '</td>'
            + '<td>' + escapeHtml(e.display_name) + ' <span class="text-muted mono" style="font-size:0.75rem">' + escapeHtml(e.node_id) + '</span></td>'
            + '<td class="mono">' + (e.credits || 0) + '</td>'
            + '<td><span class="tier-badge ' + tierClass + '">' + escapeHtml(e.tier || 'Silver') + '</span></td>'
            + '</tr>';
        }
        tbody.innerHTML = html;
      } catch (e) {
        tbody.innerHTML = '<tr><td colspan="4" class="text-muted" style="text-align:center">Error: ' + escapeHtml(e.message) + '</td></tr>';
      }
    }
  };

  // ========================================================================
  // Network Map Module — SVG world heatmap
  // ========================================================================
  var networkMap = {
    data: null,
    mapRendered: false,

    // Simplified world map: ISO alpha-2 → SVG path (low-res country outlines)
    // Paths are in equirectangular projection, viewBox 0 0 1000 500
    paths: {
      US:'M55,165L55,195L135,195L135,210L170,210L170,195L265,195L265,165Z',
      CA:'M55,90L55,165L265,165L265,105L200,90Z',
      MX:'M90,210L90,250L170,250L170,210Z',
      BR:'M260,280L260,380L360,380L360,280Z',
      AR:'M270,380L270,440L330,440L330,380Z',
      CL:'M255,340L255,450L275,450L275,340Z',
      CO:'M225,255L225,290L265,290L265,255Z',
      GB:'M440,130L440,155L458,155L458,130Z',
      FR:'M445,155L445,185L478,185L478,155Z',
      DE:'M478,140L478,170L505,170L505,140Z',
      ES:'M430,180L430,200L465,200L465,180Z',
      IT:'M478,170L478,205L498,205L498,170Z',
      NL:'M468,135L468,150L482,150L482,135Z',
      SE:'M490,80L490,135L508,135L508,80Z',
      NO:'M472,70L472,130L490,130L490,70Z',
      FI:'M510,70L510,125L530,125L530,70Z',
      PL:'M505,140L505,165L535,165L535,140Z',
      UA:'M535,140L535,170L580,170L580,140Z',
      RU:'M540,60L540,145L750,145L750,60Z',
      TR:'M540,170L540,195L590,195L590,170Z',
      IN:'M650,210L650,310L720,310L720,210Z',
      CN:'M700,130L700,230L800,230L800,130Z',
      JP:'M830,155L830,210L855,210L855,155Z',
      KR:'M810,170L810,200L830,200L830,170Z',
      AU:'M780,330L780,420L890,420L890,330Z',
      NZ:'M910,390L910,430L935,430L935,390Z',
      ZA:'M510,370L510,420L560,420L560,370Z',
      NG:'M470,275L470,305L505,305L505,275Z',
      EG:'M530,210L530,250L565,250L565,210Z',
      KE:'M555,285L555,320L580,320L580,285Z',
      SG:'M735,290L735,300L745,300L745,290Z',
      ID:'M740,290L740,330L820,330L820,290Z',
      TH:'M720,240L720,280L740,280L740,240Z',
      VN:'M740,230L740,280L755,280L755,230Z',
      PH:'M790,240L790,280L810,280L810,240Z',
      TW:'M800,215L800,235L815,235L815,215Z',
      IL:'M545,200L545,220L555,220L555,200Z',
      AE:'M600,230L600,250L625,250L625,230Z',
      SA:'M565,215L565,265L610,265L610,215Z',
      CH:'M470,162L470,175L488,175L488,162Z',
      AT:'M490,160L490,172L515,172L515,160Z',
      CZ:'M490,148L490,160L515,160L515,148Z',
      RO:'M520,160L520,178L548,178L548,160Z',
      IE:'M425,130L425,155L440,155L440,130Z',
      PT:'M420,180L420,205L432,205L432,180Z',
      DK:'M478,120L478,137L492,137L492,120Z',
      BE:'M458,148L458,162L472,162L472,148Z',
    },

    buildSvg: function() {
      var container = document.getElementById('world-map');
      if (!container) return;
      var svg = '<svg viewBox="0 0 1000 500" xmlns="http://www.w3.org/2000/svg" class="world-svg">';
      // Background
      svg += '<rect width="1000" height="500" fill="var(--bg-primary)" rx="4"/>';
      // Grid lines
      for (var x = 0; x <= 1000; x += 100) {
        svg += '<line x1="' + x + '" y1="0" x2="' + x + '" y2="500" stroke="var(--border)" stroke-width="0.3" opacity="0.5"/>';
      }
      for (var y = 0; y <= 500; y += 100) {
        svg += '<line x1="0" y1="' + y + '" x2="1000" y2="' + y + '" stroke="var(--border)" stroke-width="0.3" opacity="0.5"/>';
      }
      // Country paths
      var codes = Object.keys(networkMap.paths);
      for (var i = 0; i < codes.length; i++) {
        var code = codes[i];
        var d = networkMap.paths[code];
        svg += '<path id="region-' + code + '" d="' + d + '" fill="var(--bg-tertiary)" stroke="var(--border)" stroke-width="0.5" class="map-region" data-code="' + code + '"/>';
      }
      svg += '</svg>';
      container.innerHTML = svg;

      // Add hover tooltip handlers
      container.querySelectorAll('.map-region').forEach(function(el) {
        el.addEventListener('mouseenter', function(e) { networkMap.showTooltip(e, el.dataset.code); });
        el.addEventListener('mouseleave', function() { networkMap.hideTooltip(); });
      });

      networkMap.mapRendered = true;
    },

    refresh: async function() {
      if (!networkMap.mapRendered) networkMap.buildSvg();
      try {
        var resp = await fetch('/api/admin/network-map');
        var data = await resp.json();
        networkMap.data = data;
        networkMap.render(data);
        networkMap.populateModelFilter(data);
      } catch (e) {
        // silent
      }
    },

    render: function(data) {
      if (!data || !data.regions) return;
      var regions = data.regions;
      var filter = (document.getElementById('map-model-filter') || {}).value || '';

      // Compute counts per region
      var counts = {};
      var maxCount = 0;
      var totalNodes = 0;
      var totalRegions = 0;
      var codes = Object.keys(regions);
      for (var i = 0; i < codes.length; i++) {
        var code = codes[i];
        var r = regions[code];
        var count;
        if (filter && r.models) {
          count = r.models[filter] || 0;
        } else {
          count = r.total || 0;
        }
        if (count > 0) {
          counts[code] = count;
          totalNodes += count;
          totalRegions++;
          if (count > maxCount) maxCount = count;
        }
      }

      // Color all regions
      var allCodes = Object.keys(networkMap.paths);
      for (var j = 0; j < allCodes.length; j++) {
        var c = allCodes[j];
        var el = document.getElementById('region-' + c);
        if (!el) continue;
        var n = counts[c] || 0;
        if (n === 0) {
          el.style.fill = 'var(--bg-tertiary)';
          el.style.filter = '';
        } else {
          var intensity = Math.max(0.2, n / Math.max(maxCount, 1));
          var r2 = Math.round(20 + intensity * 20);
          var g = Math.round(70 + intensity * 60);
          var b = Math.round(180 + intensity * 75);
          el.style.fill = 'rgb(' + r2 + ',' + g + ',' + b + ')';
          el.style.filter = 'drop-shadow(0 0 ' + Math.round(intensity * 6) + 'px rgba(59,130,246,' + (intensity * 0.5).toFixed(2) + '))';
        }
      }

      document.getElementById('map-total-nodes').textContent = totalNodes;
      document.getElementById('map-total-regions').textContent = totalRegions;
      document.getElementById('map-legend-max').textContent = maxCount;
    },

    applyFilter: function() {
      if (networkMap.data) networkMap.render(networkMap.data);
    },

    populateModelFilter: function(data) {
      var sel = document.getElementById('map-model-filter');
      if (!sel || !data || !data.regions) return;
      var models = {};
      var codes = Object.keys(data.regions);
      for (var i = 0; i < codes.length; i++) {
        var r = data.regions[codes[i]];
        if (r.models) {
          var mids = Object.keys(r.models);
          for (var j = 0; j < mids.length; j++) models[mids[j]] = true;
        }
      }
      var current = sel.value;
      sel.innerHTML = '<option value="">All models</option>';
      var sorted = Object.keys(models).sort();
      for (var k = 0; k < sorted.length; k++) {
        var opt = document.createElement('option');
        opt.value = sorted[k];
        opt.textContent = sorted[k].length > 30 ? sorted[k].substring(0, 30) + '...' : sorted[k];
        if (sorted[k] === current) opt.selected = true;
        sel.appendChild(opt);
      }
    },

    updateFromWs: function(regionSummary) {
      // Quick update region counts from WebSocket without full API fetch
      if (!networkMap.mapRendered) return;
      var maxCount = 0;
      var totalNodes = 0;
      var totalRegions = 0;
      var codes = Object.keys(regionSummary);
      for (var i = 0; i < codes.length; i++) {
        var count = regionSummary[codes[i]];
        if (count > 0) {
          totalNodes += count;
          totalRegions++;
          if (count > maxCount) maxCount = count;
        }
      }
      var allCodes = Object.keys(networkMap.paths);
      for (var j = 0; j < allCodes.length; j++) {
        var c = allCodes[j];
        var el = document.getElementById('region-' + c);
        if (!el) continue;
        var n = regionSummary[c] || 0;
        if (n === 0) {
          el.style.fill = 'var(--bg-tertiary)';
          el.style.filter = '';
        } else {
          var intensity = Math.max(0.2, n / Math.max(maxCount, 1));
          var r2 = Math.round(20 + intensity * 20);
          var g = Math.round(70 + intensity * 60);
          var b = Math.round(180 + intensity * 75);
          el.style.fill = 'rgb(' + r2 + ',' + g + ',' + b + ')';
          el.style.filter = 'drop-shadow(0 0 ' + Math.round(intensity * 6) + 'px rgba(59,130,246,' + (intensity * 0.5).toFixed(2) + '))';
        }
      }
      document.getElementById('map-total-nodes').textContent = totalNodes;
      document.getElementById('map-total-regions').textContent = totalRegions;
      document.getElementById('map-legend-max').textContent = maxCount;
    },

    showTooltip: function(event, code) {
      networkMap.hideTooltip();
      var info = networkMap.data && networkMap.data.regions ? networkMap.data.regions[code] : null;
      var tip = document.createElement('div');
      tip.id = 'map-tooltip';
      tip.className = 'map-tooltip';
      var html = '<strong>' + code + '</strong>';
      if (info) {
        html += '<span class="mono" style="margin-left:8px">' + info.total + ' node' + (info.total !== 1 ? 's' : '') + '</span>';
        if (info.models) {
          var mids = Object.keys(info.models);
          if (mids.length > 0) {
            html += '<div class="mt-1" style="font-size:0.75rem">';
            for (var i = 0; i < Math.min(mids.length, 5); i++) {
              html += '<div class="flex-between" style="gap:12px"><span class="text-muted">' + escapeHtml(mids[i].length > 20 ? mids[i].substring(0, 20) + '...' : mids[i]) + '</span><span class="mono">' + info.models[mids[i]] + '</span></div>';
            }
            if (mids.length > 5) html += '<div class="text-muted">+' + (mids.length - 5) + ' more</div>';
            html += '</div>';
          }
        }
      } else {
        html += '<span class="text-muted" style="margin-left:8px">No nodes</span>';
      }
      tip.innerHTML = html;
      document.getElementById('world-map-container').appendChild(tip);
      var rect = event.target.getBoundingClientRect();
      var containerRect = document.getElementById('world-map-container').getBoundingClientRect();
      tip.style.left = Math.min(rect.left - containerRect.left + rect.width / 2, containerRect.width - 200) + 'px';
      tip.style.top = (rect.top - containerRect.top - tip.offsetHeight - 8) + 'px';
    },

    hideTooltip: function() {
      var tip = document.getElementById('map-tooltip');
      if (tip) tip.remove();
    }
  };

  // ========================================================================
  // Init
  // ========================================================================
  // Bind all UI event listeners (replaces inline onclick handlers)
  function bindEvents() {
    function on(id, event, fn) {
      var el = document.getElementById(id);
      if (el) el.addEventListener(event, fn);
    }

    // Tab buttons
    document.querySelectorAll('.tab-btn[data-tab]').forEach(function(btn) {
      btn.addEventListener('click', function() { ui.switchTab(btn.dataset.tab); });
    });

    // Setup wizard
    on('btn-prev', 'click', function() { setup.prevStep(); });
    on('btn-next', 'click', function() { setup.nextStep(); });

    // Settings modal
    on('btn-close-settings', 'click', function() { ui.closeSettings(); });
    on('btn-copy-api-key', 'click', function() { settings.copyApiKey(); });
    on('btn-save-settings', 'click', function() { settings.save(); });
    on('btn-open-settings', 'click', function() { ui.openSettings(); });

    // Model browser
    on('btn-close-model-browser', 'click', function() { ui.closeModelBrowser(); });
    on('btn-hf-search', 'click', function() { hf.search(); });
    on('hf-search-input', 'keydown', function(e) { if (e.key === 'Enter') hf.search(); });
    on('btn-open-model-browser', 'click', function() { ui.openModelBrowser(); });
    on('btn-browse-hf', 'click', function() { ui.openModelBrowser(); });
    on('link-browse-hf', 'click', function(e) { e.preventDefault(); ui.openModelBrowser(); });

    // Header
    on('hamburger-btn', 'click', function() { ui.toggleMobileSidebar(); });
    on('btn-shutdown', 'click', function() { shutdown(); });

    // Sidebar
    on('sidebar-overlay', 'click', function() { ui.toggleMobileSidebar(); });
    on('btn-new-session', 'click', function() { chat.newSession(); });
    on('btn-toggle-sidebar', 'click', function() { ui.toggleSidebar(); });

    // Chat
    on('send-btn', 'click', function() { chat.send(); });
    on('chat-input', 'keydown', function(e) { chat.handleKey(e); });

    // Network discovery
    on('btn-copy-network-code', 'click', function() { copyNetworkCode(); });
    on('btn-join-network', 'click', function() { joinNetwork(); });

    // Network map
    on('map-model-filter', 'change', function() { networkMap.applyFilter(); });
    on('btn-refresh-map', 'click', function() { networkMap.refresh(); });

    // Leaderboard
    on('btn-refresh-leaderboard', 'click', function() { identity.loadLeaderboard(); });

    // Delegated handlers for dynamically generated elements (CSP-safe)
    document.addEventListener('click', function(e) {
      var target = e.target;

      // Session delete button
      var delId = target.getAttribute('data-delete-session');
      if (delId) { e.stopPropagation(); chat.deleteSession(delId, e); return; }

      // Model action buttons
      var selectId = target.getAttribute('data-select-model');
      if (selectId) { selectModel(selectId); return; }

      var cancelId = target.getAttribute('data-cancel-download');
      if (cancelId) { cancelDownload(cancelId); return; }

      var requestId = target.getAttribute('data-request-model');
      if (requestId) { requestModel(requestId); return; }

      var removeId = target.getAttribute('data-remove-model');
      if (removeId) { removeModel(removeId); return; }

      // HF download button
      var hfRepo = target.getAttribute('data-hf-download');
      if (hfRepo) { hf.download(hfRepo, target.getAttribute('data-hf-filename') || ''); return; }
    });
  }

  function init() {
    bindEvents();

    inputEl = document.getElementById('chat-input');
    if (inputEl) {
      inputEl.addEventListener('input', autoResizeInput);
      inputEl.addEventListener('input', updateTokenCounter);
    }

    chat.loadSessions();
    chat.renderSessionList();
    chat.renderMessages();

    setup.init();
    settings.init();
    settings.loadApiKey();
    dashboard.loadInitial();
    loadModels();
    connectWebSocket();
    identity.loadNickname();

    // Start polling as fallback — will be paused once WebSocket connects
    startPolling();
  }

  // Start when DOM is ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

  // Public API
  // --- Network invite code ---
  async function loadNetworkCode() {
    try {
      var resp = await fetch('/api/admin/network-code');
      var data = await resp.json();
      var panel = document.getElementById('invite-code-panel');
      if (!panel) return;

      var phase = data.phase || 'seedling';
      var badge = document.getElementById('network-phase-badge');
      if (badge) {
        badge.textContent = phase;
        badge.className = 'badge ' + (phase === 'established' ? 'badge-green' : phase === 'growing' ? 'badge-blue' : 'badge-orange');
      }

      // Show panel when network is seedling or growing, hide when established
      if (phase === 'established') {
        panel.style.display = 'none';
      } else {
        panel.style.display = '';
        var codeInput = document.getElementById('my-network-code');
        if (codeInput && data.code) codeInput.value = data.code;
      }
    } catch (e) {}
  }

  function copyNetworkCode() {
    var input = document.getElementById('my-network-code');
    if (input && input.value) {
      navigator.clipboard.writeText(input.value).then(function() {
        ui.showBanner('success', 'Network code copied to clipboard');
      });
    }
  }

  async function joinNetwork() {
    var input = document.getElementById('join-code-input');
    var status = document.getElementById('join-status');
    if (!input || !input.value.trim()) return;

    try {
      var resp = await fetch('/api/admin/join-network', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ code: input.value.trim() })
      });
      var data = await resp.json();
      if (resp.ok) {
        if (status) status.textContent = 'Peer saved! Will connect on next discovery cycle.';
        if (status) status.style.color = 'var(--green)';
        input.value = '';
      } else {
        if (status) status.textContent = data.error || 'Failed to join';
        if (status) status.style.color = 'var(--red, #ff6464)';
      }
    } catch (e) {
      if (status) status.textContent = 'Network error';
      if (status) status.style.color = 'var(--red, #ff6464)';
    }
  }

  return {
    ui: ui,
    chat: chat,
    dashboard: dashboard,
    hf: hf,
    settings: settings,
    setup: setup,
    identity: identity,
    networkMap: networkMap,
    requestModel: requestModel,
    selectModel: selectModel,
    cancelDownload: cancelDownload,
    removeModel: removeModel,
    shutdown: shutdown,
    copyNetworkCode: copyNetworkCode,
    joinNetwork: joinNetwork,
  };
})();
