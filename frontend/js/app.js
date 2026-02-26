'use strict';

// ============================================================================
// SwarmLLM — Unified single-page application
// ============================================================================

var SwarmLLM = (function() {
  var ws = null;
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
      if (tab === 'chat') {
        chat.scrollToBottom();
        document.getElementById('chat-input').focus();
      }
    },

    toggleSidebar: function() {
      var sidebar = document.getElementById('sidebar');
      sidebar.classList.toggle('collapsed');
      var btn = sidebar.querySelector('.sidebar-toggle');
      btn.innerHTML = sidebar.classList.contains('collapsed') ? '&#9654;' : '&#9664;';
    },

    openSettings: function() {
      document.getElementById('settings-modal').classList.remove('hidden');
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
          '<button class="btn btn-ghost btn-sm session-delete" onclick="SwarmLLM.chat.deleteSession(\'' + s.id + '\', event)" title="Delete">&times;</button>';
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
        var resp = await fetch('/v1/chat/completions', {
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
  // Dashboard Module — stats, models, governance, network
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
        dashboard.renderModelsTable(models);
      } catch (e) {}

      dashboard.loadGovernanceData();
      dashboard.loadNetworkData();
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

      if (data.credits) {
        document.getElementById('credit-balance').textContent = data.credits.balance.toLocaleString();
        document.getElementById('stat-credits').textContent = data.credits.balance.toLocaleString();
        document.getElementById('credit-earned').textContent = '+' + (data.credits.lifetime_earned || 0).toLocaleString();
        document.getElementById('credit-spent').textContent = '-' + (data.credits.lifetime_spent || 0).toLocaleString();
      }
    },

    updateStats: function(data) {
      if (data.peers !== undefined) document.getElementById('stat-peers').textContent = data.peers;
      if (data.credits !== undefined) {
        var bal = typeof data.credits === 'object' ? data.credits.balance : data.credits;
        document.getElementById('stat-credits').textContent = bal.toLocaleString();
        creditHistory.push(Math.abs(bal));
        if (creditHistory.length > 30) creditHistory.shift();
        renderSparkline('credit-sparkline', creditHistory);
      }
      if (data.requests_served !== undefined) document.getElementById('stat-served').textContent = data.requests_served;
      if (data.active_requests !== undefined) document.getElementById('stat-active').textContent = data.active_requests;
    },

    renderModelsTable: function(models) {
      var table = document.getElementById('models-table');
      var empty = document.getElementById('models-empty');
      var tbody = document.getElementById('models-table-body');

      if (!models || models.length === 0) {
        table.style.display = 'none';
        empty.style.display = '';
        return;
      }

      table.style.display = '';
      empty.style.display = 'none';
      tbody.innerHTML = '';

      models.forEach(function(m) {
        var source = m.source || 'local';
        var shards = m.shards || [];
        var shardCount = m.shard_count || 0;
        var hostedShards = m.hosted_shards || 0;
        var safeId = (m.id || '').replace(/[^a-zA-Z0-9]/g, '_');

        var tr = document.createElement('tr');

        var sourceBadge = '<span class="source-badge ' + source + '">' + source + '</span>';
        if (shardCount > 1) {
          sourceBadge += ' <span class="text-muted" style="font-size:0.7rem">' + hostedShards + '/' + shardCount + ' shards</span>';
        }

        // Shard map
        var shardMap = '';
        if (shardCount > 1 && shards.length > 0) {
          shardMap = '<div class="shard-map" style="display:flex;gap:2px;margin-top:4px">';
          shards.forEach(function(s) {
            var color = s.local ? 'var(--green)' : (s.holders > 0 ? 'var(--accent)' : 'var(--border)');
            var title = 'Shard ' + s.index + ' (' + formatBytes(s.size_bytes) + ')' + (s.local ? ' - Local' : '') + (s.holders > 0 ? ' - ' + s.holders + ' holder(s)' : ' - Unavailable');
            shardMap += '<div title="' + title + '" style="width:' + Math.max(6, Math.floor(80 / shardCount)) + 'px;height:14px;border-radius:2px;background:' + color + ';cursor:help"></div>';
          });
          shardMap += '</div>';
        }

        // Availability
        var availability = '';
        if (m.local && m.status === 'loaded') {
          availability = '<span class="status-dot online"></span><span class="text-green" style="font-size:0.8rem">Loaded</span>';
        } else if (hostedShards > 0 && hostedShards === shardCount) {
          availability = '<span class="status-dot online"></span><span style="font-size:0.8rem">All shards local</span>';
        } else if (hostedShards > 0) {
          availability = '<span class="status-dot degraded"></span><span style="font-size:0.8rem">' + hostedShards + '/' + shardCount + ' shards</span>';
        } else if (m.peers_hosting > 0) {
          availability = '<span class="status-dot online"></span><span style="font-size:0.8rem">' + m.peers_hosting + ' peer' + (m.peers_hosting > 1 ? 's' : '') + '</span>';
        } else {
          availability = '<span class="text-muted" style="font-size:0.8rem">Discovered</span>';
        }
        availability += shardMap;

        // Action
        var action = '';
        if (m.status === 'loaded') {
          action = '<span class="text-green" style="font-size:0.8rem;font-weight:600">Active</span>';
        } else if (activeAcquisitions[m.id]) {
          action = '<span class="text-muted" style="font-size:0.8rem">Acquiring...</span>';
        } else if (source === 'network' || m.status === 'available' || m.status === 'partial') {
          action = '<button class="btn btn-sm btn-primary" onclick="SwarmLLM.requestModel(\'' + escapeHtml(m.id) + '\')">Download</button>';
        } else if (source === 'local' && m.status !== 'loaded') {
          action = '<span class="text-muted" style="font-size:0.8rem">Stored</span>';
        }

        var name = m.name || m.id;
        if (name.length > 40) name = name.substring(0, 40) + '...';

        tr.innerHTML = '<td><strong>' + escapeHtml(name) + '</strong></td>' +
          '<td>' + sourceBadge + '</td>' +
          '<td>' + formatBytes(m.total_size_bytes || 0) + '</td>' +
          '<td>' + availability + '</td>' +
          '<td>' + action + '</td>';
        tbody.appendChild(tr);
      });
    },

    loadGovernanceData: async function() {
      try {
        var resp = await fetch('/api/admin/governance/role');
        var role = await resp.json();
        if (role.role) document.getElementById('governance-role').textContent = capitalize(role.role);
      } catch (e) {}

      try {
        var resp = await fetch('/api/admin/proposals');
        var proposals = await resp.json();
        var list = document.getElementById('proposals-list');
        if (proposals && proposals.length > 0) {
          list.innerHTML = '';
          proposals.slice(0, 5).forEach(function(p) {
            var div = document.createElement('div');
            div.className = 'flex-between mb-1';
            div.innerHTML = '<span style="font-size:0.85rem">' + escapeHtml(p.title || 'Proposal') + '</span>' +
              '<span class="mono text-muted" style="font-size:0.8rem">' + (p.votes_for || 0) + '/' + (p.votes_against || 0) + '</span>';
            list.appendChild(div);
          });
        }
      } catch (e) {}

      try {
        var resp = await fetch('/api/admin/issues');
        var issues = await resp.json();
        var list = document.getElementById('issues-list');
        if (issues && issues.length > 0) {
          list.innerHTML = '';
          issues.slice(0, 5).forEach(function(issue) {
            var div = document.createElement('div');
            div.className = 'flex-between mb-1';
            div.innerHTML = '<span style="font-size:0.85rem">' + escapeHtml(issue.title || 'Issue') + '</span>' +
              '<span class="mono text-muted" style="font-size:0.8rem">' + (issue.upvotes || 0) + ' upvotes</span>';
            list.appendChild(div);
          });
        }
      } catch (e) {}
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

      try {
        var resp = await fetch('/api/admin/releases/latest');
        if (resp.ok) {
          var release = await resp.json();
          var el = document.getElementById('latest-release');
          el.textContent = release && release.version ? 'v' + release.version : 'No releases yet';
        }
      } catch (e) {
        document.getElementById('latest-release').textContent = 'No releases yet';
      }
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
      var safeId = modelId.replace(/[^a-zA-Z0-9]/g, '_');
      var panelId = 'acq-panel-' + safeId;
      var panel = document.getElementById(panelId);

      if (!panel) {
        var banner = document.getElementById('status-banner');
        panel = document.createElement('div');
        panel.id = panelId;
        panel.style.cssText = 'background:var(--bg-secondary);border:1px solid var(--border);border-radius:var(--radius);padding:12px 16px;margin-bottom:8px';
        banner.appendChild(panel);
      }

      if (!status) {
        panel.innerHTML = '<div style="display:flex;align-items:center;gap:8px"><div class="spinner"></div><strong>' + escapeHtml(modelId) + '</strong><span class="text-muted" style="font-size:0.8rem">Starting...</span></div>';
        return;
      }

      var state = status.state;
      var stateName = typeof state === 'string' ? state : (state && state.failed ? 'failed' : 'unknown');
      var totalBytes = status.total_bytes || 0;
      var dlBytes = status.downloaded_bytes || 0;
      var pct = totalBytes > 0 ? Math.round((dlBytes / totalBytes) * 100) : 0;
      var speed = status.speed_bytes_per_sec || 0;

      var stateColor = 'var(--accent)';
      if (stateName === 'complete') stateColor = 'var(--green)';
      else if (stateName === 'failed') stateColor = 'var(--red)';

      panel.innerHTML = '<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px">' +
        '<strong>' + escapeHtml(modelId.length > 30 ? modelId.substring(0, 30) + '...' : modelId) + '</strong>' +
        '<span class="mono" style="font-size:0.85rem">' + formatBytes(dlBytes) + ' / ' + formatBytes(totalBytes) + ' (' + pct + '%)' +
        (speed > 0 ? ' - ' + formatSpeed(speed) : '') + '</span></div>' +
        '<div style="width:100%;height:6px;background:var(--bg-tertiary);border-radius:3px;overflow:hidden">' +
        '<div style="width:' + pct + '%;height:100%;background:' + stateColor + ';transition:width 0.3s"></div></div>';
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

        results.innerHTML = '';
        data.forEach(function(model) {
          var card = document.createElement('div');
          card.className = 'hf-model-card';
          var sizeStr = model.size_bytes ? formatBytes(model.size_bytes) : 'Unknown size';
          var downloads = model.downloads ? model.downloads.toLocaleString() + ' downloads' : '';

          card.innerHTML = '<div class="hf-model-info">' +
            '<div class="hf-model-name">' + escapeHtml(model.repo_id || model.id) + '</div>' +
            '<div class="hf-model-meta">' +
            (model.filename ? '<span class="mono">' + escapeHtml(model.filename) + '</span>' : '') +
            '<span>' + sizeStr + '</span>' +
            (downloads ? '<span>' + downloads + '</span>' : '') +
            '</div>' +
            '</div>' +
            '<div class="hf-model-actions">' +
            '<select class="hf-download-mode" id="dl-mode-' + escapeHtml(model.repo_id || model.id).replace(/[^a-zA-Z0-9]/g, '_') + '">' +
            '<option value="shards">Download shards (rarest first)</option>' +
            '<option value="full">Download full model</option>' +
            '</select>' +
            '<button class="btn btn-sm btn-primary" onclick="SwarmLLM.hf.download(\'' + escapeHtml(model.repo_id || model.id) + '\', \'' + escapeHtml(model.filename || '') + '\')">Download</button>' +
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

        var resp = await fetch('/api/admin/hf/download', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ repo_id: repoId, filename: filename, mode: mode }),
        });
        var data = await resp.json();
        if (data.status === 'started' || data.status === 'acquiring') {
          ui.showBanner('success', 'Download started for ' + repoId);
          ui.closeModelBrowser();
        } else {
          ui.showBanner('warning', data.message || 'Download could not be started');
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
    save: async function() {
      var config = {
        contribution: document.getElementById('settings-contribution').value,
        max_concurrent_requests: parseInt(document.getElementById('settings-max-requests').value, 10),
        max_bandwidth_mbps: parseInt(document.getElementById('settings-bandwidth').value, 10),
        max_disk_mb: parseInt(document.getElementById('settings-disk').value, 10),
      };

      try {
        var resp = await fetch('/api/admin/config', {
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
      document.getElementById('summary-models').textContent = 'Default configuration';
    },

    submit: async function() {
      var levels = ['minimal', 'moderate', 'maximum'];
      var level = levels[parseInt(document.getElementById('contribution-slider').value, 10)];
      try {
        await fetch('/api/admin/config', {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ contribution: level }),
        });
      } catch (e) {}
      localStorage.setItem(SETUP_DONE_KEY, 'true');
      document.getElementById('setup-modal').classList.add('hidden');
    }
  };

  // ========================================================================
  // WebSocket — real-time updates
  // ========================================================================
  function connectWebSocket() {
    var protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    ws = new WebSocket(protocol + '//' + window.location.host + '/api/admin/ws');

    ws.onmessage = function(event) {
      try {
        var msg = JSON.parse(event.data);
        if (msg.type === 'stats_update') {
          dashboard.updateStats(msg.data);
          if (msg.data.acquisitions) dashboard.updateAcquisitionProgress(msg.data.acquisitions);
        }
      } catch (e) {}
    };

    ws.onclose = function() { setTimeout(connectWebSocket, 3000); };
    ws.onerror = function() { ws.close(); };
  }

  // ========================================================================
  // Model loading + selection
  // ========================================================================
  async function loadModels() {
    try {
      var resp = await fetch('/v1/models');
      var data = await resp.json();
      var sel = document.getElementById('model-select');
      sel.innerHTML = '';
      if (data.data && data.data.length > 0) {
        currentModel = data.data[0].id;
        data.data.forEach(function(m) {
          var opt = document.createElement('option');
          opt.value = m.id;
          opt.textContent = m.id.length > 30 ? m.id.substring(0, 30) + '...' : m.id;
          sel.appendChild(opt);
        });
      } else {
        sel.innerHTML = '<option value="">No model loaded</option>';
      }
    } catch (e) {}
  }

  async function requestModel(modelId) {
    try {
      var resp = await fetch('/api/admin/models/' + encodeURIComponent(modelId) + '/add', { method: 'POST' });
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

  // ========================================================================
  // Shutdown
  // ========================================================================
  async function shutdown() {
    if (!confirm('Shut down SwarmLLM node?')) return;
    try {
      await fetch('/api/admin/shutdown', { method: 'POST' });
      document.body.innerHTML = '<div style="display:flex;align-items:center;justify-content:center;height:100vh;color:var(--text-muted);font-size:1.2rem">SwarmLLM has been shut down.</div>';
    } catch (e) {
      ui.showBanner('error', 'Shutdown failed: ' + e.message);
    }
  }

  // ========================================================================
  // Helpers
  // ========================================================================
  function escapeHtml(str) {
    var div = document.createElement('div');
    div.appendChild(document.createTextNode(str));
    return div.innerHTML;
  }

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

  // ========================================================================
  // Init
  // ========================================================================
  function init() {
    inputEl = document.getElementById('chat-input');
    if (inputEl) inputEl.addEventListener('input', autoResizeInput);

    chat.loadSessions();
    chat.renderSessionList();
    chat.renderMessages();

    setup.init();
    dashboard.loadInitial();
    loadModels();
    connectWebSocket();

    setInterval(dashboard.loadInitial, 30000);
    setInterval(loadModels, 30000);
  }

  // Start when DOM is ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

  // Public API
  return {
    ui: ui,
    chat: chat,
    dashboard: dashboard,
    hf: hf,
    settings: settings,
    setup: setup,
    requestModel: requestModel,
    shutdown: shutdown,
  };
})();
