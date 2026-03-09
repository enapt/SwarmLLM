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
  // Determine initial tab from URL path for direct navigation / bookmarks
  var activeTab = (function() {
    var p = window.location.pathname;
    if (p === '/chat' || p.startsWith('/chat/')) return 'chat';
    if (p === '/admin/leaderboard') return 'leaderboard';
    if (p === '/admin/network') return 'network-map';
    if (p === '/admin/compare') return 'compare';
    return 'dashboard';
  })();

  // --- STORAGE KEYS ---
  var SESSIONS_KEY = 'swarmllm_sessions';
  var ACTIVE_SESSION_KEY = 'swarmllm_active_session';
  var SETUP_DONE_KEY = 'swarmllm_setup_done';
  var CHAT_LAYOUT_KEY = 'swarmllm_chat_layout';
  var HEALTH_INTERVAL_KEY = 'swarmllm_health_interval';

  // Provider health state: { provider: { status, latency_ms, detail, last_checked } }
  var providerHealth = {};
  var healthTimer = null;

  // Per-model availability cache: { model_id: { status, latency_ms, ts } }
  var modelStatus = {};
  var _modelStatusPending = {}; // track in-flight probes

  // Clear stale chat sessions on each page load (dev mode)
  try { localStorage.removeItem(SESSIONS_KEY); localStorage.removeItem(ACTIVE_SESSION_KEY); } catch(e) {}

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
    switchTab: function(tab, skipHistory) {
      activeTab = tab;
      // Update URL to match tab (enables bookmarks and back/forward)
      if (!skipHistory) {
        var path = tab === 'chat' ? '/chat'
          : tab === 'leaderboard' ? '/admin/leaderboard'
          : tab === 'network-map' ? '/admin/network'
          : tab === 'compare' ? '/admin/compare'
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
      // Show sidebar only on chat tab
      var sidebar = document.getElementById('sidebar');
      if (sidebar) sidebar.style.display = tab === 'chat' ? '' : 'none';
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
      if (tab === 'compare') {
        compare.loadModels();
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

    openSettings: function(scrollToProviders) {
      document.getElementById('settings-modal').classList.remove('hidden');
      settings.load();
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
      // Also show as toast for better visibility
      showToast(message, type === 'warning' ? 'warning' : type === 'error' ? 'error' : type === 'success' ? 'success' : 'info');
    }
  };

  // ========================================================================
  // Image Upload Module — paste, drag-drop, file picker for VLM
  // ========================================================================
  var pendingImages = []; // Array of { data_url: string, name: string }

  function addPendingImage(file) {
    if (!file.type.startsWith('image/')) return;
    if (pendingImages.length >= 4) {
      ui.showBanner('warning', 'Maximum 4 images per message');
      return;
    }
    var reader = new FileReader();
    reader.onload = function(e) {
      pendingImages.push({ data_url: e.target.result, name: file.name });
      renderImagePreviews();
    };
    reader.readAsDataURL(file);
  }

  function renderImagePreviews() {
    var area = document.getElementById('image-preview-area');
    if (!area) return;
    if (pendingImages.length === 0) {
      area.style.display = 'none';
      area.innerHTML = '';
      return;
    }
    area.style.display = 'flex';
    area.style.flexWrap = 'wrap';
    area.style.gap = '6px';
    area.innerHTML = '';
    pendingImages.forEach(function(img, idx) {
      var wrap = document.createElement('div');
      wrap.style.cssText = 'position:relative;display:inline-block;';
      var thumb = document.createElement('img');
      thumb.src = img.data_url;
      thumb.style.cssText = 'height:60px;max-width:100px;border-radius:6px;object-fit:cover;border:1px solid var(--border);';
      thumb.title = img.name;
      var removeBtn = document.createElement('button');
      removeBtn.textContent = '\u00D7';
      removeBtn.style.cssText = 'position:absolute;top:-4px;right:-4px;background:var(--danger);color:#fff;border:none;border-radius:50%;width:18px;height:18px;font-size:12px;cursor:pointer;line-height:18px;padding:0;';
      removeBtn.onclick = function() {
        pendingImages.splice(idx, 1);
        renderImagePreviews();
      };
      wrap.appendChild(thumb);
      wrap.appendChild(removeBtn);
      area.appendChild(wrap);
    });
  }

  function clearPendingImages() {
    pendingImages = [];
    renderImagePreviews();
  }

  function buildMessageContent(text, images) {
    if (!images || images.length === 0) return text;
    // OpenAI multimodal format: content is an array of parts
    var parts = [];
    images.forEach(function(img) {
      parts.push({
        type: 'image_url',
        image_url: { url: img.data_url }
      });
    });
    parts.push({ type: 'text', text: text || 'What is in this image?' });
    return parts;
  }

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
      // If we're already on an empty session, just reuse it (update model)
      if (currentSessionId && sessions[currentSessionId] && sessions[currentSessionId].messages.length === 0) {
        sessions[currentSessionId].model = currentModel || '';
        chat.saveSessions();
        chat.renderSessionList();
        chat.renderMessages();
        chat.updateChatHeader();
        return;
      }
      var id = 'session_' + Date.now();
      sessions[id] = { id: id, title: 'New Chat', messages: [], created: Date.now(), model: currentModel || '' };
      currentSessionId = id;
      chat.saveSessions();
      chat.renderSessionList();
      chat.renderMessages();
      chat.updateChatHeader();
      ui.switchTab('chat');
    },

    switchSession: function(id) {
      if (!sessions[id]) return;
      currentSessionId = id;
      localStorage.setItem(ACTIVE_SESSION_KEY, id);

      // Auto-select the session's model if it's still available
      var s = sessions[id];
      if (s.model) {
        var allIds = _modelDropdownData.map(function(m) { return m.id; });
        if (allIds.indexOf(s.model) !== -1) {
          selectModelDropdown(s.model, { silent: true });
        } else if (s.messages.length > 0) {
          showToast('Model "' + formatModelDisplayName(s.model) + '" is no longer available. Session is read-only until model returns.', 'warning');
        }
      }

      chat.renderSessionList();
      chat.renderMessages();
      chat.updateChatHeader();
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
        list.innerHTML = '<div class="text-muted" style="padding:12px;font-size:0.8rem">No chats yet. Type a message below to start.</div>';
        return;
      }
      list.innerHTML = '';
      sorted.forEach(function(s) {
        var div = document.createElement('div');
        div.className = 'session-item' + (s.id === currentSessionId ? ' active' : '');
        div.onclick = function() { chat.switchSession(s.id); };
        var title = s.title.length > 28 ? s.title.substring(0, 28) + '...' : s.title;
        var timeStr = '';
        if (s.created) {
          var d = new Date(s.created);
          timeStr = d.toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' });
        }
        var modelBadge = s.model ? '<span class="session-model-badge" title="' + escapeHtml(s.model) + '">' + escapeHtml(formatModelDisplayName(s.model)) + '</span>' : '';
        var metaHtml = '<span class="session-meta">' + escapeHtml(timeStr) + modelBadge + '</span>';
        var titleSpan = '<span class="session-title" data-rename-session="' + escapeHtml(s.id) + '" title="Double-click to rename">' + escapeHtml(title) + '</span>';
        div.innerHTML = '<div class="session-info">' + titleSpan + metaHtml + '</div>' +
          '<button class="btn btn-ghost btn-sm session-delete" data-delete-session="' + escapeHtml(s.id) + '" title="Delete">&times;</button>';
        list.appendChild(div);
      });
    },

    renameSession: function(id, titleEl) {
      if (!sessions[id]) return;
      var current = sessions[id].title;
      var input = document.createElement('input');
      input.type = 'text';
      input.className = 'session-title-input';
      input.value = current;
      input.maxLength = 80;
      titleEl.replaceWith(input);
      input.focus();
      input.select();
      var done = function() {
        var val = input.value.trim();
        if (val && val !== current) {
          sessions[id].title = val;
          chat.saveSessions();
        }
        chat.renderSessionList();
        chat.updateChatHeader();
      };
      input.addEventListener('blur', done);
      input.addEventListener('keydown', function(e) {
        if (e.key === 'Enter') { e.preventDefault(); input.blur(); }
        if (e.key === 'Escape') { input.value = current; input.blur(); }
      });
    },

    updateChatHeader: function() {
      var header = document.getElementById('chat-session-header');
      if (!header) return;
      if (!currentSessionId || !sessions[currentSessionId]) {
        header.classList.remove('visible');
        header.innerHTML = '';
        return;
      }
      var s = sessions[currentSessionId];
      var modelName = s.model ? formatModelDisplayName(s.model) : 'No model';
      var allIds = _modelDropdownData.map(function(m) { return m.id; });
      var available = !s.model || allIds.indexOf(s.model) !== -1;
      var badgeClass = 'chat-session-model' + (available ? '' : ' unavailable');
      var badgeTitle = available ? s.model : 'Model no longer available';
      header.classList.add('visible');
      header.innerHTML =
        '<span class="chat-session-title" id="chat-header-title" title="Click to rename">' + escapeHtml(s.title) + '</span>' +
        '<span class="' + badgeClass + '" title="' + escapeHtml(badgeTitle) + '">' + escapeHtml(modelName) + (available ? '' : ' (unavailable)') + '</span>';
    },

    renderMessages: function() {
      var container = document.getElementById('chat-messages');
      var empty = document.getElementById('chat-empty');
      container.innerHTML = '';

      chat.updateChatHeader();

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
        if (msg.images && msg.images.length > 0) {
          var html = '<div style="margin-bottom:6px;">';
          msg.images.forEach(function(url) {
            html += '<img src="' + url + '" style="max-height:120px;max-width:200px;border-radius:8px;margin-right:4px;" />';
          });
          html += '</div>' + escapeHtml(msg.content);
          appendMessageToDOM(msg.role, html, true);
        } else {
          appendMessageToDOM(msg.role, msg.content);
        }
      });
      chat.scrollToBottom();
    },

    send: async function() {
      if (isStreaming) return;
      if (!currentModel) {
        ui.showBanner('warning', 'No model available — download a model or share your Network Code to find peers');
        return;
      }

      // Check if session model is still available
      if (currentSessionId && sessions[currentSessionId] && sessions[currentSessionId].model) {
        var allIds = _modelDropdownData.map(function(m) { return m.id; });
        if (allIds.indexOf(sessions[currentSessionId].model) === -1) {
          ui.showBanner('warning', 'Model "' + formatModelDisplayName(sessions[currentSessionId].model) + '" is no longer available. Start a new session with a different model.');
          return;
        }
      }

      var input = document.getElementById('chat-input');
      var text = input.value.trim();
      var images = pendingImages.slice(); // capture before clearing
      if (!text && images.length === 0) return;

      // Ensure we have a session
      if (!currentSessionId || !sessions[currentSessionId]) {
        chat.newSession();
      }

      input.value = '';
      autoResizeInput();
      clearPendingImages();

      var session = sessions[currentSessionId];
      // Store display text and images separately for rendering
      var displayText = text || (images.length > 0 ? '[Image]' : '');
      session.messages.push({ role: 'user', content: displayText, images: images.map(function(i) { return i.data_url; }) });

      // Auto-title from first message
      if (session.messages.length === 1) {
        session.title = displayText.substring(0, 50);
        chat.renderSessionList();
      }

      chat.saveSessions();
      // Show images in chat bubble
      var userHtml = '';
      if (images.length > 0) {
        userHtml += '<div style="margin-bottom:6px;">';
        images.forEach(function(img) {
          userHtml += '<img src="' + img.data_url + '" style="max-height:120px;max-width:200px;border-radius:8px;margin-right:4px;" />';
        });
        userHtml += '</div>';
      }
      userHtml += escapeHtml(displayText);
      appendMessageToDOM('user', userHtml, true);

      // Prepare assistant message for streaming
      var assistantEl = appendMessageToDOM('assistant', '');
      var contentEl = assistantEl.querySelector('.msg-content');
      contentEl.innerHTML = '<span class="typing-indicator">Thinking...</span>';

      isStreaming = true;
      document.getElementById('send-btn').disabled = true;
      var startTime = performance.now();

      // Use session's bound model (set at creation), fall back to current selection
      var model = session.model || currentModel || 'local';
      if (!session.model) {
        session.model = model;
        chat.updateChatHeader();
        chat.renderSessionList();
      }
      // Truncate chat history to last 50 messages to prevent context overflow
      var recentMessages = session.messages.slice(-50).map(function(m) {
        if (m.images && m.images.length > 0) {
          return { role: m.role, content: buildMessageContent(m.content, m.images.map(function(url) { return { data_url: url }; })) };
        }
        return { role: m.role, content: m.content };
      });
      var body = {
        model: model,
        messages: recentMessages,
        temperature: 0.7,
        max_tokens: 2048,
        stream: true,
      };

      var fullContent = '';
      var reasoningContent = '';

      try {
        var resp = await authFetch('/v1/chat/completions', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body),
        });

        if (!resp.ok) {
          var errText = await resp.text();
          var friendlyMsg = errText;
          var hintHtml = '';
          try {
            var errJson = JSON.parse(errText);
            if (errJson.error) {
              friendlyMsg = errJson.error.message || errJson.error.detail || errText;
              if (errJson.error.hint) hintHtml = '<div class="chat-error-hint">' + escapeHtml(errJson.error.hint) + '</div>';
            }
          } catch (e) {}
          contentEl.innerHTML = escapeHtml(friendlyMsg) + hintHtml + '<div class="chat-error-actions"><button class="btn btn-sm" data-retry-chat="1">Retry</button></div>';
          contentEl.classList.add('chat-error');
          isStreaming = false;
          document.getElementById('send-btn').disabled = false;
          return;
        }

        var cleared = false;
        var reader = resp.body.getReader();
        var decoder = new TextDecoder();
        var buffer = '';
        var thinkingEl = null;

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
                // Handle reasoning_content (DeepSeek R1, reasoning models)
                if (delta.reasoning_content) {
                  if (!cleared) { contentEl.textContent = ''; cleared = true; }
                  if (!thinkingEl) {
                    thinkingEl = document.createElement('details');
                    thinkingEl.className = 'reasoning-block';
                    thinkingEl.innerHTML = '<summary>Reasoning...</summary><pre class="reasoning-content"></pre>';
                    thinkingEl.open = true;
                    contentEl.appendChild(thinkingEl);
                  }
                  reasoningContent += delta.reasoning_content;
                  thinkingEl.querySelector('.reasoning-content').textContent = reasoningContent;
                  chat.scrollToBottom();
                }
                if (delta.content) {
                  if (!cleared) { contentEl.textContent = ''; cleared = true; }
                  // Close thinking block when content starts
                  if (thinkingEl && thinkingEl.open) {
                    thinkingEl.open = false;
                    thinkingEl.querySelector('summary').textContent = 'Reasoning (' + reasoningContent.length + ' chars)';
                  }
                  fullContent += delta.content;
                  // Append text after thinking block
                  var textNode = contentEl.querySelector('.response-text');
                  if (!textNode) {
                    textNode = document.createElement('div');
                    textNode.className = 'response-text';
                    contentEl.appendChild(textNode);
                  }
                  textNode.textContent = fullContent;
                  chat.scrollToBottom();
                }
              }
            } catch (e) {}
          }
        }

        if (!cleared && !fullContent && !reasoningContent) {
          contentEl.textContent = 'No response received. The model may still be loading \u2014 try again in a moment.';
          contentEl.classList.add('chat-error');
        }
      } catch (e) {
        if (!fullContent) {
          contentEl.textContent = 'Connection failed \u2014 check that the server is running and try again.';
          contentEl.classList.add('chat-error');
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
      } catch (e) { /* config is non-critical on initial load */ }

      try {
        var resp = await fetch('/api/admin/models');
        var models = await resp.json();
        var cloudModels = [];
        try {
          var pmResp = await fetch('/api/admin/provider-models');
          if (pmResp.ok) {
            var pmData = await pmResp.json();
            cloudModels = pmData.models || [];
          }
        } catch (e) {}
        dashboard.renderModels(models, cloudModels);
      } catch (e) {
        ui.showBanner('error', 'Failed to load model list');
      }

      dlQueue.load();
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
      if (data.peers !== undefined) {
        document.getElementById('stat-peers').textContent = data.peers;
        // Show LAN peer badge if any
        var lanBadge = document.getElementById('lan-peer-badge');
        if (lanBadge) {
          if (data.lan_peers && data.lan_peers > 0) {
            lanBadge.textContent = data.lan_peers + ' LAN';
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

      // Update mode indicator on live stats changes
      updateModeIndicator(data, _cachedProviderData);
    },

    renderModels: function(models, cloudModels) {
      var list = document.getElementById('models-list');
      var empty = document.getElementById('models-empty');

      var hasCloud = cloudModels && cloudModels.length > 0;
      if ((!models || models.length === 0) && !hasCloud) {
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

        // Trust level badge
        var trustBadge = '';
        if (m.trust_level === 'network_popular') {
          trustBadge = '<span class="badge-trust badge-trust-popular" title="Widely hosted across the network">Popular</span>';
        } else if (m.trust_level === 'demand_verified') {
          trustBadge = '<span class="badge-trust badge-trust-verified" title="Has received real inference requests">Verified</span>';
        } else if (m.trust_level === 'pinned') {
          trustBadge = '<span class="badge-trust badge-trust-pinned" title="Manually approved by you">Pinned</span>';
        } else if (m.source === 'network' && hostedShards === 0) {
          trustBadge = '<span class="badge-trust badge-trust-discovered" title="Discovered via gossip — not yet verified. Auto-manage will not download unless pinned or used.">Unverified</span>';
        }

        // Meta info
        var metaParts = [];
        metaParts.push(formatBytes(m.total_size_bytes || 0));
        if (shardCount > 1) metaParts.push(shardCount + ' shards');
        if (m.estimated_vram_mb) metaParts.push('~' + formatMB(m.estimated_vram_mb) + ' VRAM');
        if (m.peers_hosting > 0) metaParts.push(m.peers_hosting + ' peer' + (m.peers_hosting !== 1 ? 's' : ''));
        else if (hostedShards > 0) metaParts.push('<span style="color:var(--orange)">Local only</span>');

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
        if (shards.length > 0) {
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

            // WI-12: Check verifying state
            if (s.download && s.download.state === 'Verifying') {
              cls = 'verifying'; dlCount++;
              label = '\u2713';
              if (missingCount > 0) missingCount--;
              if (peerCount > 0 && !s.local) peerCount--;
            }

            var title = 'Shard ' + s.index + ' (' + formatBytes(s.size_bytes) + ')';
            if (cls === 'local') title += ' \u2014 Verified, stored locally';
            else if (cls === 'peer') title += ' \u2014 Available from ' + s.holders + ' peer(s)';
            else if (cls === 'downloading') title += ' \u2014 Downloading (' + dlPct + '%)';
            else if (cls === 'verifying') title += ' \u2014 Downloaded, verifying (BLAKE3)...';
            else if (cls === 'peer-downloading') title += ' \u2014 Peer downloading (' + dlPct + '%)';
            else title += ' \u2014 Not available';

            var style = '';
            if (cls === 'downloading' || cls === 'peer-downloading') {
              style = ' style="--dl-pct:' + dlPct + '%"';
            }
            var lockIcon = s.locked ? '<span class="shard-lock-icon" title="Locked (pinned)">\uD83D\uDD12</span>' : '';
            shardHtml += '<div class="shard-cell ' + cls + (s.locked ? ' locked' : '') + '"' + style + ' data-shard="' + safeId + '-' + s.index + '" data-shard-model="' + escapeHtml(m.id) + '" data-shard-index="' + s.index + '" data-shard-locked="' + (s.locked ? '1' : '0') + '" title="' + escapeHtml(title) + '">' + label + lockIcon + '</div>';
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
          if (legendParts.length > 0) {
            // Insert legend both above (compact) and below (detailed) the grid
            shardHtml = '<div class="shard-legend shard-legend-mini" data-model-legend="' + safeId + '">' + legendParts.join('') + '</div>' + shardHtml;
          }
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

        // Probed badge — show when HF metadata fetched but no shards downloaded yet
        var probedBadge = '';
        if (m.probed && hostedShards === 0 && !isDownloading) {
          probedBadge = '<span class="badge-probed">Probed</span>';
        }

        // Gear icon for per-model auto-manage settings
        var gearHtml = '<button class="model-gear-btn" data-am-gear="' + escapeHtml(m.id) + '" title="Auto-manage settings">&#9881;</button>';

        // GGUF metadata info button (only if header file exists)
        var metaBtnHtml = '';
        if (m.has_header) {
          metaBtnHtml = '<button class="model-meta-btn" data-meta-toggle="' + escapeHtml(m.id) + '" title="GGUF Metadata">&#9432;</button>';
        }

        var name = formatModelDisplayName(m.name || m.id);

        // Unload button for loaded models (WI-3)
        var unloadHtml = '';
        if (m.status === 'loaded') {
          unloadHtml = '<button class="btn btn-sm btn-outline" data-unload-model="' + escapeHtml(m.id) + '" style="margin-right:4px">Unload</button>';
        }

        // Per-shard download bars — stacked vertically (WI-13)
        var perShardDlHtml = '';
        if (isDownloading && shards.length > 0) {
          var dlShardBars = shards.filter(function(s) {
            return s.download && s.download.state === 'Downloading';
          });
          if (dlShardBars.length > 1) {
            perShardDlHtml = '<div class="per-shard-dl">';
            dlShardBars.forEach(function(s) {
              var pct = s.download.progress_pct || 0;
              var bytes = s.download.downloaded_bytes || 0;
              var total = s.download.total_bytes || s.size_bytes || 0;
              perShardDlHtml += '<div class="per-shard-dl-row">' +
                '<span class="per-shard-dl-label">Shard ' + s.index + '</span>' +
                '<div class="per-shard-dl-bar"><div class="per-shard-dl-fill" style="width:' + pct + '%"></div></div>' +
                '<span class="per-shard-dl-pct">' + formatBytes(bytes) + '/' + formatBytes(total) + ' (' + pct + '%)</span>' +
                '</div>';
            });
            perShardDlHtml += '</div>';
          }
        }

        // Source label (HF / Network)
        var sourceLabel = '';
        if (m.source === 'network' && hostedShards === 0) {
          sourceLabel = '<span class="badge badge-remote" title="Available via network peers">Remote</span>';
        }

        card.innerHTML =
          '<div class="model-header">' +
            '<span class="model-name" title="' + escapeHtml(m.id) + '">' + escapeHtml(name) + probedBadge + sourceLabel + trustBadge + '</span>' +
            '<span>' + metaBtnHtml + gearHtml + statusHtml + (unloadHtml ? ' ' + unloadHtml : '') + (actionHtml ? ' ' + actionHtml : '') + removeHtml + '</span>' +
          '</div>' +
          '<div class="model-meta">' + metaParts.map(function(p) { return '<span>' + p + '</span>'; }).join('') + fileIndicators + '</div>' +
          shardHtml + progressHtml + perShardDlHtml +
          '<div class="gguf-metadata-panel hidden" data-meta-panel="' + escapeHtml(m.id) + '"></div>';

        list.appendChild(card);
      });

      // Cloud provider models — one compact card per provider
      if (hasCloud) {
        var providerLabels = {
          openai: 'OpenAI', anthropic: 'Anthropic', deepseek: 'DeepSeek',
          mistral: 'Mistral', groq: 'Groq', nvidia_nim: 'NVIDIA NIM',
          cerebras: 'Cerebras', sambanova: 'SambaNova', fireworks: 'Fireworks AI',
          together: 'Together AI', deepinfra: 'DeepInfra', moonshot: 'Moonshot (Kimi)'
        };
        var divider = document.createElement('div');
        divider.className = 'cloud-models-divider';
        divider.innerHTML = '<span class="cloud-divider-line"></span><span class="cloud-divider-label">\u2601\uFE0F Cloud Fallback</span><span class="cloud-divider-line"></span>';
        list.appendChild(divider);

        // Group by provider
        var byProvider = {};
        cloudModels.forEach(function(cm) {
          var p = cm.provider || 'cloud';
          if (!byProvider[p]) byProvider[p] = [];
          byProvider[p].push(cm);
        });

        // Helper: get context length from model meta
        function getCtxLen(cm) {
          if (!cm.meta) return 0;
          return cm.meta.context_length || cm.meta.context_window || cm.meta.max_model_len || 0;
        }

        // Helper: sort models by criteria
        function sortCloudModels(models, sortBy) {
          var sorted = models.slice();
          if (sortBy === 'ctx-desc') {
            sorted.sort(function(a, b) { return getCtxLen(b) - getCtxLen(a); });
          } else if (sortBy === 'ctx-asc') {
            sorted.sort(function(a, b) { return getCtxLen(a) - getCtxLen(b); });
          } else if (sortBy === 'avail') {
            // Available first (up → rate_limited → unknown → timeout/error), then by latency
            sorted.sort(function(a, b) {
              var sa = modelStatus[a.id], sb = modelStatus[b.id];
              var rank = { up: 0, rate_limited: 1, timeout: 3, unavailable: 4, not_found: 5, error: 4 };
              var ra = sa ? (rank[sa.status] !== undefined ? rank[sa.status] : 2) : 2;
              var rb = sb ? (rank[sb.status] !== undefined ? rank[sb.status] : 2) : 2;
              if (ra !== rb) return ra - rb;
              // Same rank: sort by latency (lower first), unknowns last
              var la = sa ? sa.latency_ms : 99999, lb = sb ? sb.latency_ms : 99999;
              return la - lb;
            });
          } else {
            // A-Z by name
            sorted.sort(function(a, b) {
              var na = (a.name || a.id).toLowerCase(), nb = (b.name || b.id).toLowerCase();
              return na < nb ? -1 : na > nb ? 1 : 0;
            });
          }
          return sorted;
        }

        // Helper: render model tag HTML
        function renderCloudTag(cm) {
          var metaChip = '';
          if (cm.meta) {
            var parts = [];
            if (cm.meta.owned_by) parts.push(cm.meta.owned_by);
            var ctx = getCtxLen(cm);
            if (ctx > 0) {
              var ctxK = ctx >= 1000 ? Math.round(ctx / 1000) + 'K' : ctx.toString();
              parts.push(ctxK + ' ctx');
            }
            if (parts.length > 0) metaChip = ' <span class="cloud-tag-meta">' + escapeHtml(parts.join(' \u00b7 ')) + '</span>';
          }
          var tooltip = cm.meta ? escapeHtml(cm.id + '\n' + JSON.stringify(cm.meta, null, 2)) : escapeHtml(cm.id);
          return '<span class="cloud-model-tag" data-select-cloud="' + escapeHtml(cm.id) + '" title="' + tooltip + '">' + escapeHtml(cm.name || cm.id) + metaChip + '</span>';
        }

        // Helper: render tags into container with limit
        function renderTagsInto(container, models, tagId) {
          var CLOUD_TAG_LIMIT = 12;
          var tagHtmlArr = models.map(renderCloudTag);
          var hasMore = tagHtmlArr.length > CLOUD_TAG_LIMIT;
          var visible = hasMore ? tagHtmlArr.slice(0, CLOUD_TAG_LIMIT) : tagHtmlArr;
          var hidden = hasMore ? tagHtmlArr.slice(CLOUD_TAG_LIMIT) : [];
          container.innerHTML = visible.join('') +
            (hasMore ? '<span class="cloud-tags-hidden" id="' + tagId + '" style="display:none">' + hidden.join('') + '</span>' +
              '<button class="btn btn-sm cloud-show-more" data-toggle-tags="' + tagId + '" data-show-label="Show all ' + models.length + ' models" style="margin:4px 0;font-size:0.7rem">' +
              'Show all ' + models.length + ' models</button>' : '');
        }

        Object.keys(byProvider).forEach(function(p) {
          var pLabel = providerLabels[p] || p;
          var pModels = byProvider[p];
          // Default sort: A-Z
          var sorted = sortCloudModels(pModels, 'az');
          var tagId = 'cloud-tags-' + p;
          var filterId = 'cloud-filter-' + p;
          var sortId = 'cloud-sort-' + p;

          var card = document.createElement('div');
          card.className = 'model-card cloud-model';
          card.setAttribute('data-provider', p);

          var headerHtml =
            '<div class="model-header">' +
              '<span class="model-name">' + escapeHtml(pLabel) + '</span>' +
              '<span><span class="badge badge-cloud">' + pModels.length + ' model' + (pModels.length !== 1 ? 's' : '') + '</span>' +
              '<span style="color:var(--green);font-weight:600;font-size:0.8rem;margin-left:8px">Connected</span></span>' +
            '</div>';

          // Sort + filter controls
          var controlsHtml = pModels.length > 5 ?
            '<div class="cloud-model-controls">' +
              '<input type="text" class="cloud-model-filter" id="' + filterId + '" placeholder="Filter models\u2026" autocomplete="off">' +
              '<select class="cloud-model-sort" id="' + sortId + '">' +
                '<option value="az">A\u2013Z</option>' +
                '<option value="ctx-desc">Context \u2193</option>' +
                '<option value="ctx-asc">Context \u2191</option>' +
                '<option value="avail">Availability</option>' +
              '</select>' +
            '</div>' : '';

          card.innerHTML = headerHtml + controlsHtml +
            '<div class="cloud-model-tags" id="cloud-tags-wrap-' + p + '"></div>' +
            '<div class="model-meta"><span style="color:var(--text-muted);font-size:0.75rem">Requests routed to ' + escapeHtml(pLabel) + ' API \u2014 not shared on the swarm network</span></div>';
          list.appendChild(card);

          var tagsContainer = document.getElementById('cloud-tags-wrap-' + p);
          if (tagsContainer) renderTagsInto(tagsContainer, sorted, tagId);

          // Probe visible models for availability (first 12)
          var visibleIds = sorted.slice(0, 12).map(function(cm) { return cm.id; });
          setTimeout(function() { probeModelStatus(visibleIds); }, 500);

          // Wire up filter + sort if controls exist
          if (pModels.length > 5) {
            var filterEl = document.getElementById(filterId);
            var sortEl = document.getElementById(sortId);
            function refreshTags() {
              var query = (filterEl ? filterEl.value : '').toLowerCase().trim();
              var sortBy = sortEl ? sortEl.value : 'az';
              var filtered = pModels;
              if (query) {
                filtered = pModels.filter(function(cm) {
                  var text = ((cm.name || '') + ' ' + cm.id + ' ' + (cm.meta && cm.meta.owned_by ? cm.meta.owned_by : '')).toLowerCase();
                  return text.indexOf(query) !== -1;
                });
              }
              var s = sortCloudModels(filtered, sortBy);
              renderTagsInto(tagsContainer, s, tagId + '-f');
            }
            if (filterEl) filterEl.addEventListener('input', refreshTags);
            if (sortEl) sortEl.addEventListener('change', function() {
              refreshTags();
              // When switching to availability sort, probe all visible models
              if (sortEl.value === 'avail') {
                var ids = pModels.map(function(cm) { return cm.id; });
                probeModelStatus(ids.slice(0, 20));
              }
            });
          }
        });
      }
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
            else if (sd.state === 'verifying') {
              newClass = 'verifying'; dlCount++;
              label = '\u2713';
            } else if (sd.state === 'downloading') {
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
              if (newClass === 'local') title += ' \u2014 Verified, stored locally';
              else if (newClass === 'verifying') title += ' \u2014 Downloaded, verifying (BLAKE3)...';
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
                    var segPct = sd.state === 'complete' ? 100 : (sd.progress_pct || 0);
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
            var lanTag = p.is_lan_peer ? '<span class="lan-badge">LAN</span>' : '';
            var nodeId = '<span class="mono" style="font-size:0.8rem">' + escapeHtml(p.node_id || 'unknown') + '</span>';
            var details = '';
            if (p.gpu) details += '<div style="font-size:0.75rem;color:var(--text-secondary);margin-top:3px">GPU: ' + escapeHtml(p.gpu) + '</div>';
            div.innerHTML = statusDot + lanTag + nodeId + details;
            list.appendChild(div);
          });
        }
      } catch (e) {
        var list = document.getElementById('peers-list');
        if (list) list.innerHTML = '<div class="text-muted" style="font-size:0.85rem">Failed to load peer list</div>';
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

        // WI-11: HF rate limit detection — warn if speed < 100KB/s for > 30s
        if (status.state === 'downloading' && status.source === 'huggingface') {
          var acqInfo = activeAcquisitions[modelId];
          var speed = status.speed_bytes_per_sec || 0;
          if (speed > 0 && speed < 102400) {
            if (!acqInfo._slowSince) acqInfo._slowSince = Date.now();
            else if (Date.now() - acqInfo._slowSince > 30000 && !acqInfo._throttleWarned) {
              acqInfo._throttleWarned = true;
              showToast('HuggingFace may be rate-limiting downloads. Speed: ' + formatSpeed(speed) + '. Download will continue automatically.', 'warning', 10000);
            }
          } else {
            acqInfo._slowSince = null;
          }
        }

        if (status.state === 'complete') {
          showToast('Download complete: ' + (status.model_name || modelId), 'success');
          setTimeout(function() { delete activeAcquisitions[modelId]; dashboard.loadInitial(); }, 3000);
        } else if (status.state === 'failed') {
          var reason = (typeof status.state === 'object' && status.state.failed) ? status.state.failed.reason : '';
          showToast('Download failed: ' + (status.model_name || modelId) + (reason ? ' — ' + reason : ''), 'error', 8000);
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
      if (!card) {
        // Card doesn't exist yet — trigger a refresh to create it
        loadModels();
        dashboard.loadInitial();
        return;
      }

      var stateName = typeof status.state === 'string' ? status.state : 'unknown';

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
        var resp = await fetch('/api/admin/hf/search?query=' + encodeURIComponent(query));
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
        data.forEach(function(repo) {
          var card = document.createElement('div');
          card.className = 'hf-model-card';
          var downloads = repo.downloads ? repo.downloads.toLocaleString() + ' downloads' : '';
          var likes = repo.likes ? repo.likes.toLocaleString() + ' likes' : '';
          var safeKey = (repo.repo_id || '').replace(/[^a-zA-Z0-9]/g, '_');
          var variants = repo.variants || [];
          var recommended = repo.recommended_variant || '';

          // Build variant selector
          var variantOptions = '';
          variants.forEach(function(v) {
            var sizeStr = v.size_bytes ? formatBytes(v.size_bytes) : '';
            var label = v.quant + (sizeStr ? ' \u2014 ' + sizeStr : '');
            if (v.quant === recommended) label += ' (Recommended)';
            var selected = v.quant === recommended ? ' selected' : '';
            variantOptions += '<option value="' + escapeHtml(v.filename) + '"' + selected + '>' + escapeHtml(label) + '</option>';
          });

          var fitsTag = '';
          if (repo.fits_vram === true) {
            fitsTag = '<span style="color:var(--green)" title="Fits in your GPU VRAM">Fits VRAM</span>';
          } else if (repo.fits_vram === false && variants.length > 0) {
            fitsTag = '<span style="color:var(--yellow)" title="Smallest variant may exceed your VRAM">Check VRAM</span>';
          }

          // Network replication & demand info
          var replicas = repo.network_replicas || 0;
          var networkTag = replicas > 0
            ? '<span class="badge-swarm" title="' + replicas + ' node(s) already hosting this model on the swarm">On Swarm &mdash; ' + replicas + ' node' + (replicas !== 1 ? 's' : '') + '</span>'
            : '<span class="badge-new" title="Not yet on the swarm — you will be the first node hosting this model">New to network</span>';
          var demandTag = '';
          if (replicas === 0) {
            demandTag = '<span style="color:var(--green)" title="No replicas yet — high credit earning potential">&#128176; High demand</span>';
          } else if (replicas <= 2) {
            demandTag = '<span style="color:var(--yellow)" title="Few replicas — good credit earning potential">&#128176; Medium demand</span>';
          } else {
            demandTag = '<span style="color:var(--text-muted)" title="Well replicated across the network">&#128176; Well replicated</span>';
          }

          card.innerHTML = '<div class="hf-model-info">' +
            '<div class="hf-model-name">' + escapeHtml(repo.repo_id) + '</div>' +
            '<div class="hf-model-meta">' +
            (downloads ? '<span>' + downloads + '</span>' : '') +
            (likes ? '<span>' + likes + '</span>' : '') +
            (fitsTag ? '<span>' + fitsTag + '</span>' : '') +
            '</div>' +
            '<div class="hf-model-meta">' + networkTag + demandTag + '</div>' +
            '</div>' +
            '<div class="hf-model-actions">' +
            (variants.length > 1 ? '<select class="hf-quant-select" id="quant-' + safeKey + '">' + variantOptions + '</select>' : '') +
            '<button class="btn btn-sm btn-primary" data-hf-download="' + escapeHtml(repo.repo_id) + '" data-hf-variant="' + safeKey + '">Add to node</button>' +
            '</div>';
          results.appendChild(card);
        });
      } catch (e) {
        loading.classList.add('hidden');
        results.innerHTML = '<div class="empty-state"><p>Search failed: ' + escapeHtml(e.message) + '</p></div>';
      }
    },

    download: async function(repoId, variantKey) {
      try {
        // Get selected filename from variant selector or use first variant
        var filename = '';
        if (variantKey) {
          var quantEl = document.getElementById('quant-' + variantKey);
          if (quantEl) {
            filename = quantEl.value;
          }
        }
        // Fallback: find the download button's associated filename
        if (!filename) {
          var btn = document.querySelector('[data-hf-download="' + repoId + '"]');
          filename = btn ? (btn.getAttribute('data-hf-filename') || '') : '';
        }
        if (!filename) {
          ui.showBanner('error', 'No model variant selected');
          return;
        }

        // Use peer_fair_share: backend probes internally and computes fair share
        ui.showBanner('info', 'Probing model...');
        var resp = await authFetch('/api/admin/hf/download-shards', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ repo_id: repoId, filename: filename, peer_fair_share: true }),
        });
        var data = await resp.json();
        if (!resp.ok) {
          var errMsg = (data.error && data.error.message) || 'Download failed';
          ui.showBanner('error', errMsg);
          return;
        }
        if (data.status === 'started') {
          showToast('Downloading seed shard — auto-manage will acquire more as peers join', 'success');
          ui.closeModelBrowser();
        } else {
          showToast(data.message || 'Download could not be started', 'warning');
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
      // Load saved health interval
      var healthIntervalEl = document.getElementById('settings-health-interval');
      if (healthIntervalEl) {
        try { var saved = localStorage.getItem(HEALTH_INTERVAL_KEY); if (saved) healthIntervalEl.value = saved; } catch(e) {}
      }

      // Nickname inline validation
      var nickInput = document.getElementById('settings-nickname');
      var nickError = document.getElementById('nickname-error');
      if (nickInput && nickError) {
        nickInput.addEventListener('input', function() {
          var val = nickInput.value;
          var valid = !val || /^[a-zA-Z0-9_-]+$/.test(val);
          nickError.classList.toggle('hidden', valid);
          nickInput.style.borderColor = valid ? '' : 'var(--red)';
        });
      }
      // Add show/hide toggle to provider password fields
      document.querySelectorAll('#provider-cards input[type="password"]').forEach(function(input) {
        var wrap = document.createElement('div');
        wrap.className = 'provider-key-wrap';
        wrap.style.cssText = 'position:relative;width:100%;margin-bottom:4px';
        input.parentNode.insertBefore(wrap, input);
        wrap.appendChild(input);
        input.style.marginBottom = '0';
        var toggle = document.createElement('button');
        toggle.type = 'button';
        toggle.className = 'password-toggle';
        toggle.textContent = 'Show';
        toggle.setAttribute('aria-label', 'Toggle password visibility');
        toggle.addEventListener('click', function() {
          var isPass = input.type === 'password';
          input.type = isPass ? 'text' : 'password';
          toggle.textContent = isPass ? 'Hide' : 'Show';
        });
        wrap.appendChild(toggle);
      });
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
      } catch (e) {
        ui.showBanner('error', 'Failed to load settings: ' + (e.message || 'network error'));
      }
      // Load API key and provider status
      settings.loadApiKey();
      settings.loadProviders();
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
      } catch (e) {
        ui.showBanner('error', 'Failed to load storage info');
      }
    },

    loadProviders: async function() {
      try {
        var resp = await fetch('/api/admin/providers');
        var data = await resp.json();
        if (data.providers) {
          var anyConfigured = false;
          data.providers.forEach(function(p) {
            if (p.configured) anyConfigured = true;
            var badge = document.getElementById('provider-status-' + p.name);
            if (badge) {
              if (p.configured) {
                badge.textContent = '\u2713 Active';
                badge.className = 'badge provider-badge-active';
              } else {
                badge.textContent = 'Not set';
                badge.className = 'badge';
                badge.style.color = '';
              }
            }
            // Highlight the provider card
            var card = badge && badge.closest('.provider-card');
            if (card) {
              if (p.configured) {
                card.classList.add('provider-active');
              } else {
                card.classList.remove('provider-active');
              }
            }
          });
          // Auto-expand cloud providers section if none configured (first-run UX)
          if (!anyConfigured) {
            var section = document.getElementById('settings-providers-section');
            if (section) section.open = true;
          }
        }
      } catch (e) {
        ui.showBanner('error', 'Failed to load provider status');
      }
    },

    saveProviders: async function() {
      var keys = {};
      ['anthropic', 'openai', 'deepseek', 'mistral', 'groq', 'nvidia_nim', 'cerebras', 'sambanova', 'fireworks', 'together', 'deepinfra', 'moonshot'].forEach(function(name) {
        var input = document.getElementById('provider-key-' + name);
        if (input && input.value) {
          keys[name + '_key'] = input.value;
        }
      });
      if (Object.keys(keys).length === 0) return;
      try {
        await authFetch('/api/admin/providers', {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(keys),
        });
        // Clear inputs after save and refresh status
        ['anthropic', 'openai', 'deepseek', 'mistral', 'groq', 'nvidia_nim', 'cerebras', 'sambanova', 'fireworks', 'together', 'deepinfra', 'moonshot'].forEach(function(name) {
          var input = document.getElementById('provider-key-' + name);
          if (input) input.value = '';
        });
        settings.loadProviders();
        loadModels();
        loadModeIndicator();
        ui.showBanner('success', 'Provider keys saved');
      } catch (e) {
        ui.showBanner('error', 'Failed to save provider keys: ' + (e.message || 'network error'));
      }
    },

    testProvider: async function(name) {
      var input = document.getElementById('provider-key-' + name);
      var badge = document.getElementById('provider-status-' + name);
      if (!input) return;
      var key = input.value;
      if (!key) {
        ui.showBanner('error', 'Enter an API key first');
        return;
      }
      badge.textContent = 'Testing...';
      badge.className = 'badge badge-testing';
      try {
        // Save the key first, then test
        var saveBody = {};
        saveBody[name + '_key'] = key;
        await authFetch('/api/admin/providers', {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(saveBody),
        });
        // Test by making a minimal request
        var testResp;
        if (name === 'anthropic') {
          testResp = await authFetch('/v1/messages', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ model: 'claude-haiku-4-5-20251001', max_tokens: 1, messages: [{ role: 'user', content: 'hi' }] }),
          });
        } else {
          var modelMap = { openai: 'gpt-4o-mini', deepseek: 'deepseek-chat', mistral: 'mistral-small-latest', groq: 'llama-3.1-8b-instant', nvidia_nim: 'meta/llama-3.1-8b-instruct', cerebras: 'cerebras:llama-3.1-8b', sambanova: 'sambanova:Meta-Llama-3.3-70B-Instruct', fireworks: 'accounts/fireworks/models/llama-v3p3-70b-instruct', together: 'together:meta-llama/Llama-3.3-70B-Instruct-Turbo', deepinfra: 'deepinfra:meta-llama/Llama-3.3-70B-Instruct', moonshot: 'moonshot-v1-8k' };
          testResp = await authFetch('/v1/chat/completions', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ model: modelMap[name] || name + '-test', max_tokens: 1, messages: [{ role: 'user', content: 'hi' }] }),
          });
        }
        if (testResp.ok) {
          badge.textContent = '\u2713 Active';
          badge.className = 'badge provider-badge-active';
          ui.showBanner('success', name + ' API key verified');
          var testCard = badge.closest('.provider-card');
          if (testCard) testCard.classList.add('provider-active');
          loadModels();
          loadModeIndicator();
        } else {
          var err = await testResp.text();
          var friendlyErr = err;
          try { var ej = JSON.parse(err); friendlyErr = (ej.error && ej.error.message) || err; } catch(pe) {}
          badge.textContent = '\u2717 Failed';
          badge.className = 'badge badge-error';
          ui.showBanner('error', name + ' test failed: ' + friendlyErr);
        }
        input.value = '';
      } catch (e) {
        badge.textContent = '\u2717 Error';
        badge.className = 'badge badge-error';
        ui.showBanner('error', name + ' test failed: ' + e.message);
      }
    },

    save: async function() {
      var saveBtn = document.getElementById('btn-save-settings');
      if (saveBtn) { saveBtn.disabled = true; saveBtn.textContent = 'Saving...'; }

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

      // Save health check interval to localStorage and restart polling
      var healthIntervalEl = document.getElementById('settings-health-interval');
      if (healthIntervalEl) {
        try { localStorage.setItem(HEALTH_INTERVAL_KEY, healthIntervalEl.value); } catch(e) {}
        startHealthPolling();
      }

      // Save nickname if provided
      await identity.saveNickname();
      // Save provider keys if any were entered
      await settings.saveProviders();

      if (saveBtn) { saveBtn.disabled = false; saveBtn.textContent = 'Save Settings'; }
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
          'Low impact: <5% CPU, limited storage. Best for shared or low-spec machines.',
          'Balanced: ~25% CPU, moderate storage. Good for most users.',
          'Full power: 75%+ CPU, 50%+ storage. Best for dedicated nodes.',
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
          indicator.setAttribute('aria-selected', 'true');
        } else if (i < setup.currentStep) {
          body.classList.add('hidden');
          indicator.classList.remove('active');
          indicator.classList.add('done');
          indicator.setAttribute('aria-selected', 'false');
        } else {
          body.classList.add('hidden');
          indicator.classList.remove('active', 'done');
          indicator.setAttribute('aria-selected', 'false');
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
            '<p style="margin-bottom:8px">No models on this node yet.</p>' +
            '<p class="text-muted" style="font-size:0.85rem"><strong>Three ways to get started:</strong><br>' +
            '1. Download models from HuggingFace using <strong>Browse Models</strong> on the dashboard<br>' +
            '2. Share your Network Code to find peers who already have models<br>' +
            '3. Add a cloud provider API key for instant access</p>' +
            '<button class="btn btn-sm" id="setup-add-provider-btn" style="margin-top:10px;font-size:0.8rem">Add Cloud Provider Key (optional)</button>' +
            '</div>';
          var provBtn = document.getElementById('setup-add-provider-btn');
          if (provBtn) provBtn.onclick = function() {
            document.getElementById('setup-wizard').style.display = 'none';
            ui.openSettings();
            var section = document.getElementById('settings-providers-section');
            if (section) { section.open = true; section.scrollIntoView({behavior:'smooth'}); }
          };
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
      var nick = (document.getElementById('setup-nickname').value || '').trim();
      document.getElementById('summary-nickname').textContent = nick || 'Anonymous';
      var levels = ['minimal', 'moderate', 'maximum'];
      var val = parseInt(document.getElementById('contribution-slider').value, 10);
      document.getElementById('summary-contribution').textContent = capitalize(levels[val]);
      document.getElementById('summary-gpu').textContent = setup.hwData && setup.hwData.gpu_name ? setup.hwData.gpu_name : 'CPU only';
      document.getElementById('summary-ram').textContent = formatMB(setup.hwData ? setup.hwData.total_ram_mb || 0 : 0);
      document.getElementById('summary-disk').textContent = formatMB(setup.hwData ? setup.hwData.available_disk_mb || 0 : 0);
      var autoManage = document.getElementById('setup-auto-manage').checked;
      document.getElementById('summary-auto-manage').textContent = autoManage ? 'Enabled' : 'Disabled';
      var provNames = {openai:'OpenAI',deepseek:'DeepSeek',groq:'Groq',nvidia_nim:'NVIDIA NIM',cerebras:'Cerebras',sambanova:'SambaNova',anthropic:'Anthropic',mistral:'Mistral',fireworks:'Fireworks',together:'Together',deepinfra:'DeepInfra'};
      document.getElementById('summary-provider').textContent = setup._savedProvider ? provNames[setup._savedProvider] || setup._savedProvider : 'None (can add later)';
      document.getElementById('summary-models').textContent = 'Default configuration';
    },

    submit: async function() {
      var levels = ['minimal', 'moderate', 'maximum'];
      var level = levels[parseInt(document.getElementById('contribution-slider').value, 10)];
      var autoManage = document.getElementById('setup-auto-manage').checked;
      try {
        var resp = await authFetch('/api/admin/config', {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            contribution: level,
            auto_manage_shards: autoManage,
          }),
        });
        if (!resp.ok) {
          ui.showBanner('error', 'Setup failed — could not save configuration');
          return;
        }
      } catch (e) {
        ui.showBanner('error', 'Setup failed: ' + (e.message || 'network error'));
        return;
      }
      // Save nickname if set
      var nick = (document.getElementById('setup-nickname').value || '').trim();
      if (nick) {
        try {
          await authFetch('/api/identity/nickname', {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ nickname: nick, visibility: 'nickname' }),
          });
        } catch (e) { /* non-critical */ }
      }
      localStorage.setItem(SETUP_DONE_KEY, 'true');
      // Also persist to server so other clients / restarts see setup as done
      try {
        await authFetch('/api/admin/config', {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ setup_done: true }),
        });
      } catch (e) { /* non-critical */ }
      document.getElementById('setup-modal').classList.add('hidden');
      ui.showBanner('success', 'Setup complete! Welcome to SwarmLLM.');
    },

    finish: function() {
      localStorage.setItem(SETUP_DONE_KEY, 'true');
      document.getElementById('setup-modal').classList.add('hidden');
      ui.showBanner('info', 'Setup skipped — you can configure everything in Settings.');
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
          dlQueue.updateFromWs(msg.data.acquisitions);
          updateChatDownloadProgress(msg.data.acquisitions);
          if (msg.data.region_summary && activeTab === 'network-map') {
            networkMap.updateFromWs(msg.data.region_summary);
          }
        } else if (msg.type === 'lan_peer_discovered') {
          var count = msg.data.peer_count || 1;
          showLanDiscoveryToast('Found ' + count + ' peer' + (count !== 1 ? 's' : '') + ' on your local network \u2014 zero configuration needed!');
        } else if (msg.type === 'update_available') {
          showUpdateBanner(msg.data);
        } else if (msg.type === 'prune_event') {
          var d = msg.data;
          var freed = formatBytes(d.freed_bytes || 0);
          var text = 'Pruned shard ' + d.shard_index + ' of ' + (d.model_name || d.model_id) +
            ' \u2014 ' + d.holder_count_before + '\u2192' + d.holder_count_after + ' holders (freed ' + freed + ')';
          showPruneToast(text);
          // models_changed event from prune will trigger refresh below
        } else if (msg.type === 'models_changed') {
          // Debounce: coalesce rapid model change events
          if (window._modelsChangedTimer) clearTimeout(window._modelsChangedTimer);
          window._modelsChangedTimer = setTimeout(function() {
            loadModels();
            loadModeIndicator();
            dashboard.loadInitial();
          }, 1000);
        }
      } catch (e) {
        // WS parse error — ignore malformed frames
      }
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

  // Provider health probe — lightweight ping to each configured provider
  function getHealthInterval() {
    try { var v = parseInt(localStorage.getItem(HEALTH_INTERVAL_KEY)); return v > 0 ? v : 30; } catch(e) { return 30; }
  }

  async function fetchProviderHealth() {
    try {
      var resp = await authFetch('/api/admin/provider-health');
      if (!resp.ok) return;
      var data = await resp.json();
      var now = Date.now();
      (data.providers || []).forEach(function(p) {
        providerHealth[p.provider] = {
          status: p.status,
          latency_ms: p.latency_ms,
          detail: p.detail || '',
          last_checked: now
        };
      });
      updateProviderHealthBadges();
    } catch (e) { /* health probe is non-critical */ }
  }

  function startHealthPolling() {
    if (healthTimer) clearInterval(healthTimer);
    var intervalSec = getHealthInterval();
    if (intervalSec <= 0) return; // disabled
    fetchProviderHealth(); // immediate first check
    healthTimer = setInterval(fetchProviderHealth, intervalSec * 1000);
  }

  // --- Provider badge icons (inline SVG, 18x18) ---
  var providerIcons = {
    anthropic: '<svg viewBox="0 0 18 18" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M10.5 2L5 16h2.5l1.2-3h4.6l1.2 3H17L11.5 2h-1zm.5 3.5L13.5 12h-5L11 5.5z" fill="currentColor"/></svg>',
    openai: '<svg viewBox="0 0 18 18" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M9 1.5a7.5 7.5 0 100 15 7.5 7.5 0 000-15zm0 2l3.9 2.25v4.5L9 12.5 5.1 10.25v-4.5L9 3.5zm0 1.73L6.6 6.75v3.5L9 11.77l2.4-1.52v-3.5L9 5.23z" fill="currentColor"/></svg>',
    deepseek: '<svg viewBox="0 0 18 18" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M9 2C5.5 2 3 5 3 8c0 2.5 1.5 4 3 5l1 3h4l1-3c1.5-1 3-2.5 3-5 0-3-2.5-6-6-6zm-1 6.5a1 1 0 11-2 0 1 1 0 012 0zm4 0a1 1 0 11-2 0 1 1 0 012 0z" fill="currentColor"/></svg>',
    mistral: '<svg viewBox="0 0 18 18" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M2 5h14M2 9h14M2 13h14" stroke="currentColor" stroke-width="2" stroke-linecap="round"/><circle cx="5" cy="5" r="1.2" fill="currentColor"/><circle cx="13" cy="9" r="1.2" fill="currentColor"/><circle cx="8" cy="13" r="1.2" fill="currentColor"/></svg>',
    groq: '<svg viewBox="0 0 18 18" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M9 2l1.5 5H16l-4 3.5 1.5 5.5L9 12.5 4.5 16 6 10.5 2 7h5.5L9 2z" fill="currentColor"/></svg>',
    nvidia_nim: '<svg viewBox="0 0 18 18" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M3 14V7l3-3 3 3v2l3-3 3 3v5" stroke="#76b900" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" fill="none"/></svg>',
    cerebras: '<svg viewBox="0 0 18 18" fill="none" xmlns="http://www.w3.org/2000/svg"><rect x="3" y="3" width="12" height="12" rx="2" stroke="currentColor" stroke-width="1.5" fill="none"/><rect x="6" y="6" width="6" height="6" rx="1" stroke="currentColor" stroke-width="1" fill="none"/><circle cx="9" cy="9" r="1.5" fill="currentColor"/></svg>',
    sambanova: '<svg viewBox="0 0 18 18" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M2 12c2-4 4-6 7-6s5 2 7 6" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" fill="none"/><path d="M2 9c2-3 4-5 7-5s5 2 7 5" stroke="currentColor" stroke-width="1.2" stroke-linecap="round" fill="none" opacity="0.5"/></svg>',
    fireworks: '<svg viewBox="0 0 18 18" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M9 2v4M9 12v4M2 9h4M12 9h4M4 4l3 3M11 11l3 3M14 4l-3 3M7 11l-3 3" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/><circle cx="9" cy="9" r="1.5" fill="currentColor"/></svg>',
    together: '<svg viewBox="0 0 18 18" fill="none" xmlns="http://www.w3.org/2000/svg"><circle cx="5" cy="5" r="2" stroke="currentColor" stroke-width="1.3" fill="none"/><circle cx="13" cy="5" r="2" stroke="currentColor" stroke-width="1.3" fill="none"/><circle cx="9" cy="13" r="2" stroke="currentColor" stroke-width="1.3" fill="none"/><path d="M6.5 6.5L8 11.5M11.5 6.5L10 11.5M7 5h4" stroke="currentColor" stroke-width="1" stroke-linecap="round"/></svg>',
    deepinfra: '<svg viewBox="0 0 18 18" fill="none" xmlns="http://www.w3.org/2000/svg"><rect x="4" y="3" width="10" height="4" rx="1" stroke="currentColor" stroke-width="1.3" fill="none"/><rect x="4" y="11" width="10" height="4" rx="1" stroke="currentColor" stroke-width="1.3" fill="none"/><circle cx="6.5" cy="5" r="0.8" fill="currentColor"/><circle cx="6.5" cy="13" r="0.8" fill="currentColor"/><path d="M9 7v4" stroke="currentColor" stroke-width="1.2" stroke-linecap="round"/></svg>',
    moonshot: '<svg viewBox="0 0 18 18" fill="none" xmlns="http://www.w3.org/2000/svg"><circle cx="9" cy="9" r="6" stroke="currentColor" stroke-width="1.3" fill="none"/><path d="M11 6a4 4 0 0 0-4 6" stroke="currentColor" stroke-width="1.3" stroke-linecap="round"/><circle cx="7" cy="8" r="0.8" fill="currentColor"/></svg>'
  };

  var providerDisplayNames = {
    anthropic: 'Anthropic', openai: 'OpenAI', deepseek: 'DeepSeek',
    mistral: 'Mistral', groq: 'Groq', nvidia_nim: 'NVIDIA NIM',
    cerebras: 'Cerebras', sambanova: 'SambaNova', fireworks: 'Fireworks',
    together: 'Together', deepinfra: 'DeepInfra', moonshot: 'Kimi'
  };

  function updateProviderBannerBadges() {
    var strip = document.getElementById('provider-badges');
    if (!strip) return;
    var configured = Object.keys(providerHealth);
    if (configured.length === 0) {
      strip.classList.add('hidden');
      return;
    }
    strip.classList.remove('hidden');
    strip.innerHTML = '';
    configured.sort().forEach(function(p) {
      var h = providerHealth[p];
      var badge = document.createElement('div');
      badge.className = 'provider-badge' + (h.status === 'up' ? ' badge-active' : '');
      var dotClass = 'dot-down';
      var latencyText = '';
      if (h.status === 'up') {
        dotClass = h.latency_ms < 500 ? 'dot-fast' : h.latency_ms < 2000 ? 'dot-ok' : 'dot-slow';
        latencyText = h.latency_ms + 'ms';
      } else if (h.status === 'rate_limited') {
        dotClass = 'dot-ok';
        latencyText = 'Limited';
      } else if (h.status === 'timeout') {
        dotClass = 'dot-slow';
        latencyText = 'Timeout';
      } else if (h.status === 'auth_error') {
        latencyText = 'Auth err';
      } else if (h.status === 'overloaded') {
        dotClass = 'dot-ok';
        latencyText = 'Busy';
      } else {
        latencyText = 'Down';
      }
      var iconHtml = providerIcons[p] || '';
      var name = providerDisplayNames[p] || p;
      badge.innerHTML = '<span class="pb-icon">' + iconHtml + '</span>' +
        '<span class="pb-name">' + escapeHtml(name) + '</span>' +
        '<span class="pb-dot ' + dotClass + '"></span>' +
        (latencyText ? '<span class="pb-latency">' + escapeHtml(latencyText) + '</span>' : '');
      badge.title = name + ': ' + h.status + (h.detail ? ' — ' + h.detail : '') + (h.latency_ms ? ' (' + h.latency_ms + 'ms)' : '');
      strip.appendChild(badge);
    });
  }

  function updateProviderHealthBadges() {
    // Update top banner badges
    updateProviderBannerBadges();
    // Update dashboard provider cards
    Object.keys(providerHealth).forEach(function(p) {
      var h = providerHealth[p];
      var badge = document.getElementById('health-badge-' + p);
      if (!badge) {
        // Find the provider card header and append badge
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
      } else if (h.status === 'rate_limited') {
        statusIcon = 'Rate limited';
        statusClass = 'health-warn';
      } else if (h.status === 'timeout') {
        statusIcon = 'Timeout';
        statusClass = 'health-down';
      } else if (h.status === 'auth_error') {
        statusIcon = 'Auth error';
        statusClass = 'health-down';
      } else if (h.status === 'overloaded') {
        statusIcon = 'Overloaded';
        statusClass = 'health-warn';
      } else {
        statusIcon = 'Error';
        statusClass = 'health-down';
      }
      badge.className = 'provider-health-badge ' + statusClass;
      badge.textContent = statusIcon;
      badge.title = h.status + (h.detail ? ': ' + h.detail : '') + ' (' + h.latency_ms + 'ms)';
    });

    // Update model dropdown items with latency hints
    Object.keys(providerHealth).forEach(function(p) {
      var h = providerHealth[p];
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
  }

  // Probe individual model availability (batched, max 20 at a time)
  function probeModelStatus(modelIds) {
    // Filter out already-probed (within 60s) and in-flight models
    var now = Date.now();
    var toProbe = modelIds.filter(function(id) {
      if (_modelStatusPending[id]) return false;
      var cached = modelStatus[id];
      if (cached && (now - cached.ts) < 60000) return false;
      return true;
    });
    if (toProbe.length === 0) return;
    toProbe = toProbe.slice(0, 20);
    toProbe.forEach(function(id) { _modelStatusPending[id] = true; });

    authFetch('/api/admin/provider-model-status', {
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
        modelStatus[m.model] = { status: m.status, latency_ms: m.latency_ms, ts: ts };
        delete _modelStatusPending[m.model];
      });
      updateModelStatusBadges();
    }).catch(function() {
      toProbe.forEach(function(id) { delete _modelStatusPending[id]; });
    });
  }

  function modelStatusBadgeHtml(modelId) {
    var s = modelStatus[modelId];
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
  }

  function updateModelStatusBadges() {
    // Update cloud model tags on dashboard
    document.querySelectorAll('.cloud-model-tag[data-select-cloud]').forEach(function(tag) {
      var modelId = tag.getAttribute('data-select-cloud');
      var existing = tag.querySelector('.model-status-badge');
      var html = modelStatusBadgeHtml(modelId);
      if (html) {
        if (existing) { existing.outerHTML = html; } else { tag.insertAdjacentHTML('beforeend', ' ' + html); }
      }
    });
    // Update chat dropdown items
    document.querySelectorAll('.model-dropdown-item[data-value]').forEach(function(item) {
      var modelId = item.getAttribute('data-value');
      var existing = item.querySelector('.model-status-badge');
      var html = modelStatusBadgeHtml(modelId);
      if (html) {
        if (existing) { existing.outerHTML = html; } else { item.insertAdjacentHTML('beforeend', ' ' + html); }
      }
    });
  }

  // ========================================================================
  // Model loading + selection
  // ========================================================================
  var _modelDropdownData = []; // [{id, name, group, provider}]

  async function loadModels() {
    try {
      // Fetch admin model list + provider models in parallel
      var adminResp = await fetch('/api/admin/models');
      var adminModels = adminResp.ok ? await adminResp.json() : [];

      var providerModels = [];
      try {
        var pmResp = await fetch('/api/admin/provider-models');
        if (pmResp.ok) {
          var pmData = await pmResp.json();
          providerModels = pmData.models || [];
        }
      } catch (e) {}

      // Build set of ready model IDs
      var readySet = {};
      adminModels.forEach(function(m) {
        var isReady = m.status === 'loaded' || m.status === 'ready' ||
          (m.global_available === m.shard_count && m.shard_count > 0);
        if (isReady) readySet[m.id] = true;
      });

      var readyModels = adminModels.filter(function(m) { return readySet[m.id]; });
      var hasAny = readyModels.length > 0 || providerModels.length > 0;

      // Build grouped data
      var providerLabels = {
        openai: 'OpenAI', anthropic: 'Anthropic', deepseek: 'DeepSeek',
        mistral: 'Mistral', groq: 'Groq', nvidia_nim: 'NVIDIA NIM',
        cerebras: 'Cerebras', sambanova: 'SambaNova', fireworks: 'Fireworks AI',
        together: 'Together AI', deepinfra: 'DeepInfra', moonshot: 'Moonshot (Kimi)'
      };
      var groups = [];
      _modelDropdownData = [];

      if (readyModels.length > 0) {
        var items = readyModels.map(function(m) {
          var displayName = formatModelDisplayName(m.name || m.id);
          return { id: m.id, name: displayName.length > 40 ? displayName.substring(0, 40) + '...' : displayName, group: 'local' };
        });
        groups.push({ key: 'local', label: 'Local / Network', items: items });
        _modelDropdownData = _modelDropdownData.concat(items);
      }

      if (providerModels.length > 0) {
        var byProvider = {};
        providerModels.forEach(function(m) {
          var p = m.provider || 'cloud';
          if (!byProvider[p]) byProvider[p] = [];
          byProvider[p].push(m);
        });
        Object.keys(byProvider).forEach(function(p) {
          var items = byProvider[p].map(function(m) {
            var item = { id: m.id, name: m.name || m.id, group: p, provider: p };
            if (m.meta) item.meta = m.meta;
            return item;
          });
          // Sort A-Z by name for consistent ordering
          items.sort(function(a, b) {
            var na = a.name.toLowerCase(), nb = b.name.toLowerCase();
            return na < nb ? -1 : na > nb ? 1 : 0;
          });
          groups.push({ key: p, label: (providerLabels[p] || p) + ' (cloud)', items: items });
          _modelDropdownData = _modelDropdownData.concat(items);
        });
      }

      // Render custom dropdown
      renderModelDropdown(groups, hasAny);

      // Restore saved selection — prioritize current session's model
      if (hasAny) {
        var allIds = _modelDropdownData.map(function(m) { return m.id; });
        var sessionModel = currentSessionId && sessions[currentSessionId] ? sessions[currentSessionId].model : null;
        var savedModel = null;
        try { savedModel = localStorage.getItem('swarmllm_current_model'); } catch (e) {}
        var preferred = sessionModel || savedModel;
        var found = preferred && allIds.indexOf(preferred) !== -1;
        selectModelDropdown(found ? preferred : allIds[0], { silent: true });
      } else {
        currentModel = '';
        updateModelDropdownLabel('Select model...');
      }

      syncMobileModelSelect();
      updateChatAvailability(hasAny);
    } catch (e) {
      ui.showBanner('error', 'Failed to load models: ' + (e.message || 'network error'));
    }
  }

  function renderModelDropdown(groups, hasAny) {
    var list = document.getElementById('model-dropdown-list');
    if (!list) return;
    list.innerHTML = '';

    if (!hasAny) {
      list.innerHTML = '<div class="model-dropdown-empty">No models available<br><span style="font-size:0.72rem;color:var(--text-muted)">Download a model, find peers via Network Code, or add a cloud provider</span></div>';
      return;
    }

    groups.forEach(function(g) {
      var groupEl = document.createElement('div');
      groupEl.className = 'model-dropdown-group';
      groupEl.setAttribute('data-group', g.key);

      var header = document.createElement('div');
      header.className = 'model-dropdown-group-header';
      header.innerHTML = '<span class="group-arrow">&#9662;</span> ' + escapeHtml(g.label) + ' <span style="opacity:0.5;font-weight:400">(' + g.items.length + ')</span>';
      header.addEventListener('click', function() {
        groupEl.classList.toggle('collapsed');
      });
      groupEl.appendChild(header);

      var itemsEl = document.createElement('div');
      itemsEl.className = 'model-dropdown-group-items';
      g.items.forEach(function(item) {
        var el = document.createElement('div');
        el.className = 'model-dropdown-item';
        el.setAttribute('data-value', item.id);
        el.setAttribute('data-search', (item.name + ' ' + item.id).toLowerCase());
        // Build display: name + optional meta chips
        var nameSpan = document.createElement('span');
        nameSpan.textContent = item.name;
        el.appendChild(nameSpan);
        if (item.meta) {
          var metaParts = [];
          var m = item.meta;
          // Extract useful fields from provider metadata
          if (m.owned_by) metaParts.push(m.owned_by);
          if (m.context_length || m.context_window) metaParts.push((m.context_length || m.context_window).toLocaleString() + ' ctx');
          if (m.max_tokens) metaParts.push(m.max_tokens.toLocaleString() + ' max');
          if (m.pricing) {
            var p = m.pricing;
            if (p.prompt !== undefined) metaParts.push('$' + p.prompt + '/1K in');
            if (p.completion !== undefined) metaParts.push('$' + p.completion + '/1K out');
          }
          if (m.status && m.status !== 'available') metaParts.push(m.status);
          if (metaParts.length > 0) {
            var metaSpan = document.createElement('span');
            metaSpan.className = 'model-meta-chips';
            metaSpan.style.cssText = 'font-size:0.7rem;opacity:0.5;margin-left:6px';
            metaSpan.textContent = metaParts.join(' · ');
            el.appendChild(metaSpan);
          }
          // Full meta as tooltip
          el.title = item.id + '\n' + JSON.stringify(item.meta, null, 2);
        } else {
          el.title = item.id;
        }
        el.addEventListener('click', function() {
          selectModelDropdown(item.id);
          closeModelDropdown();
        });
        itemsEl.appendChild(el);
      });
      groupEl.appendChild(itemsEl);
      list.appendChild(groupEl);
    });
  }

  function selectModelDropdown(modelId, opts) {
    opts = opts || {};
    var prevModel = currentModel;
    currentModel = modelId;
    document.getElementById('model-select').value = modelId;
    try { localStorage.setItem('swarmllm_current_model', modelId); } catch (e) {}

    // Update trigger label
    var item = _modelDropdownData.find(function(m) { return m.id === modelId; });
    updateModelDropdownLabel(item ? item.name : modelId);

    // Update selected state
    var items = document.querySelectorAll('#model-dropdown-list .model-dropdown-item');
    items.forEach(function(el) {
      el.classList.toggle('selected', el.getAttribute('data-value') === modelId);
    });

    // Flash the trigger to confirm selection
    var trigger = document.getElementById('model-dropdown-trigger');
    if (trigger) {
      trigger.classList.remove('flash');
      void trigger.offsetWidth; // force reflow to restart animation
      trigger.classList.add('flash');
    }

    // If model changed while in a session with messages, auto-start new session
    if (!opts.silent && prevModel && prevModel !== modelId && currentSessionId && sessions[currentSessionId]) {
      var s = sessions[currentSessionId];
      if (s.messages.length > 0) {
        chat.newSession();
        showToast('New session started for ' + formatModelDisplayName(modelId), 'info');
      } else {
        // Empty session — just update its model
        s.model = modelId;
        chat.saveSessions();
        chat.updateChatHeader();
        chat.renderSessionList();
      }
    }
  }

  function updateModelDropdownLabel(text) {
    var label = document.getElementById('model-dropdown-label');
    if (label) label.textContent = text;
  }

  function closeModelDropdown() {
    var dd = document.getElementById('model-dropdown');
    if (dd) dd.classList.remove('open');
  }

  function initModelDropdown() {
    var trigger = document.getElementById('model-dropdown-trigger');
    var dd = document.getElementById('model-dropdown');
    var search = document.getElementById('model-dropdown-search');
    if (!trigger || !dd) return;

    trigger.addEventListener('click', function(e) {
      e.stopPropagation();
      dd.classList.toggle('open');
      if (dd.classList.contains('open') && search) {
        search.value = '';
        filterModelDropdown('');
        setTimeout(function() { search.focus(); }, 50);
      }
    });

    // Filter on search input
    if (search) {
      search.addEventListener('input', function() {
        filterModelDropdown(search.value);
      });
      search.addEventListener('keydown', function(e) {
        if (e.key === 'Escape') { closeModelDropdown(); }
        if (e.key === 'Enter') {
          // Select first visible item
          var first = document.querySelector('#model-dropdown-list .model-dropdown-item:not(.hidden)');
          if (first) {
            selectModelDropdown(first.getAttribute('data-value'));
            closeModelDropdown();
          }
        }
      });
    }

    // Close on outside click
    document.addEventListener('click', function(e) {
      if (!dd.contains(e.target)) closeModelDropdown();
    });
  }

  function filterModelDropdown(query) {
    var q = query.toLowerCase().trim();
    var items = document.querySelectorAll('#model-dropdown-list .model-dropdown-item');
    items.forEach(function(el) {
      var match = !q || el.getAttribute('data-search').indexOf(q) !== -1;
      el.classList.toggle('hidden', !match);
    });
    // Auto-expand groups with matches, collapse empty ones
    var groups = document.querySelectorAll('#model-dropdown-list .model-dropdown-group');
    groups.forEach(function(g) {
      var visibleItems = g.querySelectorAll('.model-dropdown-item:not(.hidden)');
      if (q) {
        g.classList.toggle('collapsed', visibleItems.length === 0);
      }
    });
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
    selectModelDropdown(modelId);
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
  // ========================================================================
  // Shard Context Menu
  // ========================================================================
  // ========================================================================
  // Unified Toast Notification System
  // ========================================================================
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
    var toast = document.createElement('div');
    toast.className = 'toast toast-' + type;
    var icons = { success: '\u2713', error: '\u2717', warning: '\u26A0', info: '\u2139' };
    toast.innerHTML = '<span class="toast-icon">' + (icons[type] || icons.info) + '</span>' +
      '<span class="toast-text">' + escapeHtml(text) + '</span>' +
      '<button class="toast-close" onclick="this.parentNode.remove()">\u00d7</button>';
    container.appendChild(toast);
    requestAnimationFrame(function() { toast.classList.add('toast-show'); });
    var timer = setTimeout(function() {
      toast.classList.remove('toast-show');
      setTimeout(function() { toast.remove(); }, 300);
    }, duration);
    toast.addEventListener('click', function() { clearTimeout(timer); toast.remove(); });
  }

  function showLanDiscoveryToast(text) {
    showToast(text, 'success', 8000);
  }

  // ========================================================================
  // Update Banner
  // ========================================================================
  function showUpdateBanner(data) {
    // Only show once — don't re-create if already visible
    if (document.getElementById('update-banner')) return;
    var banner = document.createElement('div');
    banner.id = 'update-banner';
    banner.style.cssText = 'position:fixed;top:0;left:0;right:0;z-index:10000;background:#f9e2af;color:#1e1e2e;padding:0.6rem 1rem;display:flex;align-items:center;justify-content:center;gap:1rem;font-size:0.85rem;font-weight:500;box-shadow:0 2px 8px rgba(0,0,0,0.3)';
    var text = 'Update available: v' + escapeHtml(data.current_version) + ' \u2192 v' + escapeHtml(data.latest_version);
    banner.innerHTML = '<span>' + text + '</span>';
    if (data.downloaded) {
      var applyBtn = document.createElement('button');
      applyBtn.textContent = 'Apply & Restart';
      applyBtn.style.cssText = 'background:#1e1e2e;color:#f9e2af;border:none;border-radius:4px;padding:0.3rem 0.8rem;cursor:pointer;font-size:0.8rem;font-weight:600';
      applyBtn.onclick = async function() {
        applyBtn.disabled = true;
        applyBtn.textContent = 'Applying...';
        try {
          var resp = await authFetch('/api/admin/update/apply', { method: 'POST' });
          if (resp.ok) {
            banner.querySelector('span').textContent = 'Update applied! Restart the daemon to use v' + escapeHtml(data.latest_version);
            applyBtn.style.display = 'none';
          } else {
            var err = await resp.json().catch(function() { return {}; });
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
      dlBtn.style.cssText = 'background:#1e1e2e;color:#f9e2af;border:none;border-radius:4px;padding:0.3rem 0.8rem;cursor:pointer;font-size:0.8rem;font-weight:600';
      dlBtn.onclick = async function() {
        dlBtn.disabled = true;
        dlBtn.textContent = 'Checking...';
        try {
          var resp = await authFetch('/api/admin/update/check', { method: 'POST' });
          if (resp.ok) {
            var result = await resp.json();
            if (result.status === 'update_available' && result.info && result.info.downloaded) {
              dlBtn.textContent = 'Applying...';
              var applyResp = await authFetch('/api/admin/update/apply', { method: 'POST' });
              if (applyResp.ok) {
                banner.querySelector('span').textContent = 'Update applied! Restart the daemon to use v' + escapeHtml(data.latest_version);
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

  function showPruneToast(text) {
    showToast(text, 'info', 6000);
  }

  // ========================================================================
  // Prune History Card
  // ========================================================================
  async function loadPruneHistory() {
    try {
      var resp = await authFetch('/api/admin/prune-history');
      if (!resp.ok) return;
      var data = await resp.json();
      renderPruneHistory(data.events || []);
    } catch (e) {
      var el = document.getElementById('prune-history-list');
      if (el) el.innerHTML = '<div class="text-muted" style="padding:0.5rem">Could not load prune history</div>';
    }
  }

  function renderPruneHistory(events) {
    var el = document.getElementById('prune-history-list');
    if (!el) return;
    if (events.length === 0) {
      el.innerHTML = '<div class="text-muted" style="padding:0.5rem">No prune events yet</div>';
      return;
    }
    var html = '';
    events.slice(0, 20).forEach(function(e) {
      var freed = formatBytes(e.freed_bytes || 0);
      var ts = e.timestamp ? new Date(e.timestamp).toLocaleString() : '';
      html += '<div class="prune-event-row" style="display:flex;justify-content:space-between;padding:0.3rem 0;border-bottom:1px solid var(--border,#313244);font-size:0.75rem">' +
        '<span>' + escapeHtml(e.model_name || e.model_id) + ' shard ' + escapeHtml(String(e.shard_index)) + '</span>' +
        '<span class="text-muted">' + escapeHtml(freed) + ' \u2022 ' + escapeHtml(String(e.holder_count_before)) + '\u2192' + escapeHtml(String(e.holder_count_after)) + ' \u2022 ' + escapeHtml(ts) + '</span>' +
      '</div>';
    });
    el.innerHTML = html;
  }

  // ========================================================================
  // Resource Schedule Card
  // ========================================================================
  async function loadSchedule() {
    var el = document.getElementById('schedule-form');
    try {
      var resp = await authFetch('/api/admin/schedule');
      if (!resp.ok) {
        if (el) el.innerHTML = '<div class="text-muted" style="font-size:0.85rem">No schedule configured</div>';
        return;
      }
      var s = await resp.json();
      renderScheduleCard(s);
    } catch (e) {
      if (el) el.innerHTML = '<div class="text-muted" style="font-size:0.85rem">No schedule configured</div>';
    }
  }

  function renderScheduleCard(s) {
    var el = document.getElementById('schedule-form');
    if (!el) return;
    el.innerHTML =
      '<div class="am-row"><label><input type="checkbox" id="sched-enabled"' + (s.enabled ? ' checked' : '') + '> Enable reduced hours</label></div>' +
      '<div class="am-row"><label>Start hour (0-23):</label> <input type="number" id="sched-start" value="' + (s.reduced_hours_start || 22) + '" min="0" max="23" style="width:3rem"></div>' +
      '<div class="am-row"><label>End hour (0-23):</label> <input type="number" id="sched-end" value="' + (s.reduced_hours_end || 8) + '" min="0" max="23" style="width:3rem"></div>' +
      '<div class="am-row"><label>Contribution:</label> <select id="sched-contrib"><option value="minimal"' + (s.reduced_contribution === 'minimal' ? ' selected' : '') + '>Minimal</option><option value="moderate"' + (s.reduced_contribution === 'moderate' ? ' selected' : '') + '>Moderate</option></select></div>' +
      '<div class="am-row"><label>Prune aggressiveness:</label> <select id="sched-prune-agg"><option value="conservative"' + (s.prune_aggressiveness === 'conservative' ? ' selected' : '') + '>Conservative</option><option value="normal"' + (s.prune_aggressiveness === 'normal' ? ' selected' : '') + '>Normal</option><option value="aggressive"' + (s.prune_aggressiveness === 'aggressive' ? ' selected' : '') + '>Aggressive</option></select></div>' +
      '<div class="am-row"><button class="btn btn-sm btn-primary" id="sched-save-btn">Save Schedule</button></div>';
    var saveBtn = document.getElementById('sched-save-btn');
    if (saveBtn) saveBtn.addEventListener('click', saveSchedule);
  }

  async function saveSchedule() {
    try {
      var resp = await authFetch('/api/admin/schedule', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          enabled: document.getElementById('sched-enabled').checked,
          reduced_hours_start: parseInt(document.getElementById('sched-start').value, 10) || 22,
          reduced_hours_end: parseInt(document.getElementById('sched-end').value, 10) || 8,
          reduced_contribution: document.getElementById('sched-contrib').value,
          prune_aggressiveness: document.getElementById('sched-prune-agg').value,
        }),
      });
      if (resp.ok) {
        ui.showBanner('success', 'Resource schedule saved');
      } else {
        var err = await resp.json().catch(function() { return {}; });
        ui.showBanner('error', err.error ? err.error.message : 'Save failed');
      }
    } catch (e) {
      ui.showBanner('error', 'Save failed: ' + e.message);
    }
  }

  var shardMenu = {
    menu: null,
    currentModel: null,
    currentIndex: null,
    currentState: null,

    init: function() {
      this.menu = document.getElementById('shard-context-menu');
    },

    show: function(modelId, shardIndex, shardState, x, y, isLocked) {
      if (!this.menu) this.init();
      this.currentModel = modelId;
      this.currentIndex = shardIndex;
      this.currentState = shardState;
      this.currentLocked = !!isLocked;

      var header = document.getElementById('shard-ctx-header');
      var btn = document.getElementById('shard-ctx-action');
      header.textContent = 'Shard ' + shardIndex;

      if (shardState === 'local') {
        btn.textContent = 'Remove this shard';
        btn.className = 'shard-ctx-btn danger';
      } else if (shardState === 'downloading') {
        btn.textContent = 'Cancel download';
        btn.className = 'shard-ctx-btn danger';
      } else {
        btn.textContent = 'Download this shard';
        btn.className = 'shard-ctx-btn';
      }

      // Lock/unlock button
      var lockBtn = document.getElementById('shard-ctx-lock');
      if (!lockBtn) {
        lockBtn = document.createElement('button');
        lockBtn.id = 'shard-ctx-lock';
        lockBtn.className = 'shard-ctx-btn';
        lockBtn.addEventListener('click', function() { shardMenu.toggleLock(); });
        btn.parentNode.insertBefore(lockBtn, btn.nextSibling);
      }
      lockBtn.textContent = isLocked ? 'Unlock shard' : 'Lock shard (pin)';
      lockBtn.style.display = (shardState === 'local') ? '' : 'none';

      // Position menu at click, clamped to viewport
      var mw = 180, mh = 100;
      var left = Math.min(x, window.innerWidth - mw - 8);
      var top = Math.min(y, window.innerHeight - mh - 8);
      this.menu.style.left = left + 'px';
      this.menu.style.top = top + 'px';
      this.menu.style.display = '';
    },

    hide: function() {
      if (this.menu) this.menu.style.display = 'none';
    },

    execute: async function() {
      var modelId = this.currentModel;
      var idx = this.currentIndex;
      var state = this.currentState;
      this.hide();

      if (state === 'local') {
        // Remove single shard
        if (!confirm('Remove shard ' + idx + ' of ' + modelId + '?')) return;
        try {
          var resp = await authFetch('/api/admin/models/' + encodeURIComponent(modelId) + '/shards/' + idx, { method: 'DELETE' });
          if (resp.ok) {
            ui.showBanner('success', 'Shard ' + idx + ' removed');
            loadModels();
          } else {
            var errData = await resp.json().catch(function() { return {}; });
            ui.showBanner('error', errData.error ? errData.error.message : 'Failed to remove shard');
          }
        } catch (e) {
          ui.showBanner('error', 'Remove failed: ' + e.message);
        }
      } else if (state === 'downloading') {
        // Cancel download for this model
        cancelDownload(modelId);
      } else {
        // Download single shard — look up HF source first
        try {
          var srcResp = await fetch('/api/admin/hf/source/' + encodeURIComponent(modelId));
          if (!srcResp.ok) {
            ui.showBanner('error', 'No HuggingFace source found for this model');
            return;
          }
          var src = await srcResp.json();
          var dlResp = await authFetch('/api/admin/hf/download-shards', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ repo_id: src.repo_id, filename: src.filename, shards: [idx], model_id: modelId }),
          });
          if (dlResp.ok) {
            ui.showBanner('success', 'Downloading shard ' + idx);
            loadModels();
          } else {
            var errData2 = await dlResp.json().catch(function() { return {}; });
            ui.showBanner('error', errData2.error ? errData2.error.message : 'Download failed');
          }
        } catch (e) {
          ui.showBanner('error', 'Download failed: ' + e.message);
        }
      }
    },

    toggleLock: async function() {
      var modelId = this.currentModel;
      var idx = this.currentIndex;
      var newLocked = !this.currentLocked;
      this.hide();
      try {
        var resp = await authFetch('/api/admin/models/' + encodeURIComponent(modelId) + '/shards/' + idx + '/lock', {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ locked: newLocked }),
        });
        if (resp.ok) {
          ui.showBanner('success', 'Shard ' + idx + (newLocked ? ' locked' : ' unlocked'));
          loadModels();
        } else {
          ui.showBanner('error', 'Failed to update shard lock');
        }
      } catch (e) {
        ui.showBanner('error', 'Lock update failed: ' + e.message);
      }
    }
  };

  // ========================================================================
  // GGUF Metadata Panel
  // ========================================================================
  var metadataCache = {};

  async function toggleMetadataPanel(modelId) {
    var panel = document.querySelector('[data-meta-panel="' + modelId + '"]');
    if (!panel) return;
    if (!panel.classList.contains('hidden')) { panel.classList.add('hidden'); return; }
    panel.classList.remove('hidden');
    if (panel.innerHTML) return;

    panel.innerHTML = '<div class="meta-loading"><span class="spinner" style="width:14px;height:14px;border-width:1.5px"></span> Loading metadata...</div>';
    try {
      var data = metadataCache[modelId];
      if (!data) {
        var resp = await authFetch('/api/admin/models/' + encodeURIComponent(modelId) + '/metadata');
        if (!resp.ok) throw new Error('Failed to load metadata');
        data = await resp.json();
        metadataCache[modelId] = data;
      }
      renderMetadataPanel(panel, data);
    } catch (e) {
      panel.innerHTML = '<div class="meta-error">Failed to load GGUF metadata</div>';
    }
  }

  function renderMetadataPanel(panel, data) {
    var html = '<div class="meta-header">GGUF Metadata</div>';
    var g = data.general || {};
    var m = data.model || {};
    var summaryParts = [];
    if (g.architecture) {
      var archTag = '<span class="meta-tag">' + escapeHtml(g.architecture) + '</span>';
      if (g.architecture_supported === false) {
        archTag += '<span class="meta-tag" style="background:var(--error-bg,#5c2020);color:var(--error-fg,#ff6b6b)">unsupported</span>';
      }
      summaryParts.push(archTag);
    }
    if (g.quantization) summaryParts.push('<span class="meta-tag">' + escapeHtml(g.quantization) + '</span>');
    if (m.context_length) summaryParts.push('<span class="meta-tag">ctx ' + m.context_length.toLocaleString() + '</span>');
    if (m.block_count) summaryParts.push('<span class="meta-tag">' + m.block_count + ' layers</span>');
    if (m.vocab_size) summaryParts.push('<span class="meta-tag">vocab ' + m.vocab_size.toLocaleString() + '</span>');
    if (summaryParts.length > 0) html += '<div class="meta-summary">' + summaryParts.join('') + '</div>';

    html += '<table class="meta-table"><thead><tr><th colspan="2">Model Parameters</th></tr></thead><tbody>';
    var modelFields = [
      ['Context Length', m.context_length], ['Layers (block_count)', m.block_count],
      ['Embedding Dimension', m.embedding_length], ['Attention Heads', m.head_count],
      ['KV Heads (GQA)', m.head_count_kv], ['RoPE Dimension', m.rope_dimension_count],
      ['RoPE Freq Base', m.rope_freq_base], ['RMS Norm Epsilon', m.layer_norm_rms_epsilon],
      ['Vocab Size', m.vocab_size],
    ];
    modelFields.forEach(function(f) {
      if (f[1] != null) {
        var val = typeof f[1] === 'number' ? f[1].toLocaleString() : escapeHtml(String(f[1]));
        html += '<tr><td class="meta-key">' + f[0] + '</td><td class="meta-val">' + val + '</td></tr>';
      }
    });
    html += '</tbody></table>';

    var t = data.tokenizer || {};
    if (t.model || t.eos_token_id != null || t.bos_token_id != null) {
      html += '<table class="meta-table"><thead><tr><th colspan="2">Tokenizer</th></tr></thead><tbody>';
      [['Tokenizer Model', t.model], ['Pre-tokenizer', t.pre], ['BOS Token ID', t.bos_token_id],
       ['EOS Token ID', t.eos_token_id], ['Padding Token ID', t.padding_token_id]
      ].forEach(function(f) {
        if (f[1] != null) html += '<tr><td class="meta-key">' + escapeHtml(f[0]) + '</td><td class="meta-val">' + escapeHtml(String(f[1])) + '</td></tr>';
      });
      html += '</tbody></table>';
    }

    var tens = data.tensors || {};
    if (tens.count) html += '<div class="meta-tensor-info">' + tens.count + ' tensors, data offset: ' + formatBytes(tens.data_offset || 0) + '</div>';

    var raw = data.raw || [];
    if (raw.length > 0) {
      html += '<details class="meta-raw-details"><summary>All metadata keys (' + raw.length + ')</summary>';
      html += '<table class="meta-table meta-raw-table"><tbody>';
      raw.forEach(function(r) { html += '<tr><td class="meta-key">' + escapeHtml(r.key) + '</td><td class="meta-val">' + escapeHtml(r.value) + '</td></tr>'; });
      html += '</tbody></table></details>';
    }
    panel.innerHTML = html;
  }

  // ========================================================================
  // Download Queue
  // ========================================================================
  var dlQueue = {
    load: async function() {
      try {
        var resp = await authFetch('/api/admin/downloads');
        if (!resp.ok) return;
        var data = await resp.json();
        dlQueue.render(data.downloads || []);
      } catch (e) {
        // Download queue is non-critical — silent
      }
    },

    render: function(downloads) {
      var panel = document.getElementById('download-queue-panel');
      var list = document.getElementById('download-queue-list');
      var empty = document.getElementById('download-queue-empty');
      var count = document.getElementById('download-queue-count');
      if (!panel || !list) return;

      var active = downloads.filter(function(d) {
        var st = typeof d.state === 'string' ? d.state : '';
        return st !== 'complete';
      });

      if (active.length === 0 && downloads.length === 0) { panel.classList.add('hidden'); return; }
      panel.classList.remove('hidden');
      if (active.length === 0) { list.innerHTML = ''; empty.classList.remove('hidden'); count.textContent = ''; return; }

      empty.classList.add('hidden');
      count.textContent = active.length + ' active';
      list.innerHTML = '';

      active.forEach(function(dl) {
        var item = document.createElement('div');
        item.className = 'dl-queue-item';
        item.setAttribute('data-dl-model', dl.model_id);

        var stateName = typeof dl.state === 'string' ? dl.state : 'unknown';
        var stateLabel = stateName, stateClass = 'waiting';
        if (stateName === 'downloading') { stateLabel = 'Downloading'; stateClass = 'active'; }
        else if (stateName === 'awaiting_manifest') { stateLabel = 'Awaiting manifest'; stateClass = 'waiting'; }
        else if (stateName === 'complete') { stateLabel = 'Complete'; stateClass = 'done'; }
        else if (stateName.indexOf('failed') >= 0 || typeof dl.state === 'object') {
          stateLabel = 'Failed'; stateClass = 'fail';
          if (typeof dl.state === 'object' && dl.state.failed) stateLabel = 'Failed: ' + escapeHtml((dl.state.failed.reason || '').substring(0, 40));
        }

        var sourceLabel = dl.source === 'huggingface' ? 'HF' : 'Network';
        var sourceClass = dl.source === 'huggingface' ? 'hf' : 'net';
        var pct = dl.overall_pct || 0;
        var speed = dl.speed_bytes_per_sec || 0;
        var etaStr = '';
        if (dl.eta_secs) {
          var s = dl.eta_secs;
          if (s >= 3600) etaStr = Math.floor(s / 3600) + 'h ' + Math.floor((s % 3600) / 60) + 'm';
          else if (s >= 60) etaStr = Math.floor(s / 60) + 'm ' + (s % 60) + 's';
          else etaStr = s + 's';
        }

        var statsRight = formatBytes(dl.downloaded_bytes || 0) + ' / ' + formatBytes(dl.total_bytes || 0);
        if (speed > 0) statsRight += ' \u00b7 ' + formatSpeed(speed);
        if (etaStr) statsRight += ' \u00b7 ETA ' + etaStr;

        var cancelBtn = dl.cancellable ? '<button class="dl-queue-cancel" data-dl-cancel="' + escapeHtml(dl.model_id) + '">Cancel</button>' : '';
        var logToggle = (dl.log && dl.log.length > 0) ? '<button class="dl-queue-log-toggle" data-dl-log-toggle="' + escapeHtml(dl.model_id) + '">Log (' + dl.log.length + ')</button>' : '';
        var logHtml = '';
        if (dl.log && dl.log.length > 0) {
          logHtml = '<div class="dl-queue-log" data-dl-log="' + escapeHtml(dl.model_id) + '">' +
            dl.log.map(function(l) { return '<div class="dl-queue-log-line">' + escapeHtml(l) + '</div>'; }).join('') + '</div>';
        }

        var shardInfo = dl.downloaded_shards + '/' + dl.total_shards + ' shards';
        if (dl.verified_shards > 0) shardInfo += ' (' + dl.verified_shards + ' verified)';

        item.innerHTML =
          '<div class="dl-queue-row">' +
            '<span class="dl-queue-name" title="' + escapeHtml(dl.model_id) + '">' + escapeHtml(dl.model_name || dl.model_id) + '</span>' +
            '<div class="dl-queue-actions">' +
              '<span class="dl-queue-source ' + sourceClass + '">' + sourceLabel + '</span>' +
              '<span class="dl-queue-state ' + stateClass + '">' + stateLabel + '</span>' +
              cancelBtn +
            '</div>' +
          '</div>' +
          '<div class="dl-queue-bar"><div class="dl-queue-bar-fill" style="width:' + pct + '%"></div></div>' +
          '<div class="dl-queue-stats">' +
            '<span>' + shardInfo + ' \u00b7 ' + pct + '%</span>' +
            '<span>' + statsRight + '</span>' +
          '</div>' +
          '<div class="dl-queue-row">' + logToggle + '</div>' + logHtml;

        list.appendChild(item);
      });
    },

    updateFromWs: function(acquisitions) {
      if (!acquisitions || acquisitions.length === 0) return;
      var panel = document.getElementById('download-queue-panel');
      if (!panel) return;

      var hasActive = acquisitions.some(function(a) {
        var st = typeof a.state === 'string' ? a.state : '';
        return st === 'downloading' || st === 'awaiting_manifest';
      });
      if (hasActive && panel.classList.contains('hidden')) { dlQueue.load(); return; }

      acquisitions.forEach(function(acq) {
        var item = document.querySelector('[data-dl-model="' + acq.model_id + '"]');
        if (!item) {
          if (acq.state === 'downloading' || acq.state === 'awaiting_manifest') dlQueue.load();
          return;
        }

        var totalBytes = acq.total_bytes || 0;
        var dlBytes = acq.downloaded_bytes || 0;
        var pct = totalBytes > 0 ? Math.min(100, Math.round((dlBytes / totalBytes) * 100)) : 0;
        var speed = acq.speed_bytes_per_sec || 0;

        var barFill = item.querySelector('.dl-queue-bar-fill');
        if (barFill) barFill.style.width = pct + '%';

        var statsEl = item.querySelector('.dl-queue-stats');
        if (statsEl) {
          var shardInfo = (acq.downloaded_shards || 0) + '/' + (acq.total_shards || 0) + ' shards';
          var right = formatBytes(dlBytes) + ' / ' + formatBytes(totalBytes);
          if (speed > 0) right += ' \u00b7 ' + formatSpeed(speed);
          if (speed > 0 && totalBytes > dlBytes) {
            var etaSecs = Math.round((totalBytes - dlBytes) / speed);
            var etaStr;
            if (etaSecs >= 3600) etaStr = Math.floor(etaSecs / 3600) + 'h ' + Math.floor((etaSecs % 3600) / 60) + 'm';
            else if (etaSecs >= 60) etaStr = Math.floor(etaSecs / 60) + 'm ' + (etaSecs % 60) + 's';
            else etaStr = etaSecs + 's';
            right += ' \u00b7 ETA ' + etaStr;
          }
          statsEl.innerHTML = '<span>' + shardInfo + ' \u00b7 ' + pct + '%</span><span>' + right + '</span>';
        }

        if (typeof acq.state === 'string' && acq.state === 'complete') {
          setTimeout(function() { dlQueue.load(); }, 2000);
        }
      });
    },

    cancelDownload: async function(modelId) {
      try {
        var resp = await authFetch('/api/admin/downloads/' + encodeURIComponent(modelId) + '/cancel', { method: 'POST' });
        if (resp.ok) {
          ui.showBanner('success', 'Download cancelled');
          setTimeout(function() { dlQueue.load(); loadModels(); }, 1000);
        } else {
          var err = await resp.json().catch(function() { return {}; });
          ui.showBanner('error', err.error || 'Failed to cancel download');
        }
      } catch (e) {
        ui.showBanner('error', 'Cancel failed: ' + e.message);
      }
    }
  };

  // ========================================================================
  // Per-Model Auto-Manage Panel
  // ========================================================================
  async function toggleAutoManagePanel(modelId) {
    // Find model card
    var card = document.querySelector('[data-model-id="' + modelId + '"]');
    if (!card) return;

    // If panel already open, close it
    var existing = card.querySelector('.auto-manage-panel');
    if (existing) { existing.remove(); return; }

    // Fetch current policy
    var policy = { enabled: true, max_shards: 0, prune_enabled: true };
    try {
      var resp = await fetch('/api/admin/models/' + encodeURIComponent(modelId) + '/auto-manage');
      if (resp.ok) policy = await resp.json();
    } catch (e) {
      ui.showBanner('error', 'Could not load auto-manage policy');
    }

    var panel = document.createElement('div');
    panel.className = 'auto-manage-panel';
    panel.innerHTML =
      '<div class="am-row">' +
        '<label><input type="checkbox" id="am-enabled-' + escapeHtml(modelId) + '"' + (policy.enabled ? ' checked' : '') + '> Auto-manage enabled</label>' +
      '</div>' +
      '<div class="am-row">' +
        '<label><input type="checkbox" id="am-prune-' + escapeHtml(modelId) + '"' + (policy.prune_enabled !== false ? ' checked' : '') + '> Auto-prune enabled</label>' +
      '</div>' +
      '<div class="am-row">' +
        '<label>Max shards:</label>' +
        '<input type="number" id="am-max-' + escapeHtml(modelId) + '" value="' + (policy.max_shards || 0) + '" min="0" step="1">' +
        '<span class="text-muted" style="font-size:0.7rem">0 = unlimited</span>' +
      '</div>' +
      '<div class="am-row">' +
        '<button class="btn btn-sm btn-primary" data-am-save="' + escapeHtml(modelId) + '">Save</button>' +
      '</div>';
    card.appendChild(panel);
  }

  async function saveAutoManagePolicy(modelId) {
    var safeId = escapeHtml(modelId);
    var enabledEl = document.getElementById('am-enabled-' + safeId);
    var maxEl = document.getElementById('am-max-' + safeId);
    var pruneEl = document.getElementById('am-prune-' + safeId);
    if (!enabledEl || !maxEl) return;

    try {
      var resp = await authFetch('/api/admin/models/' + encodeURIComponent(modelId) + '/auto-manage', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          enabled: enabledEl.checked,
          max_shards: parseInt(maxEl.value, 10) || 0,
          prune_enabled: pruneEl ? pruneEl.checked : true,
        }),
      });
      if (resp.ok) {
        ui.showBanner('success', 'Auto-manage policy saved');
        // Close panel
        var card = document.querySelector('[data-model-id="' + modelId + '"]');
        var panel = card ? card.querySelector('.auto-manage-panel') : null;
        if (panel) panel.remove();
      } else {
        var errData = await resp.json().catch(function() { return {}; });
        ui.showBanner('error', errData.error ? errData.error.message : 'Save failed');
      }
    } catch (e) {
      ui.showBanner('error', 'Save failed: ' + e.message);
    }
  }

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
  // Unload Model (WI-3)
  // ========================================================================
  async function unloadModel(modelId) {
    try {
      var resp = await authFetch('/api/admin/models/' + encodeURIComponent(modelId) + '/unload', { method: 'POST' });
      if (resp.ok) {
        showToast('Model unloaded: ' + formatModelDisplayName(modelId), 'success');
        loadModels();
      } else {
        var errData = await resp.json().catch(function() { return {}; });
        showToast(errData.error || 'Failed to unload model', 'error');
      }
    } catch (e) {
      showToast('Unload failed: ' + e.message, 'error');
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
    // Don't render when all values are zero (looks like meaningless dashes)
    var hasActivity = data.some(function(v) { return v !== 0; });
    if (!hasActivity) { container.innerHTML = '<span class="text-muted" style="font-size:0.7rem">Credit activity will appear here</span>'; return; }
    var min = Math.min.apply(null, data);
    var max = Math.max.apply(null, data);
    var range = (max - min) || 1;
    container.innerHTML = '';
    data.forEach(function(val) {
      var bar = document.createElement('div');
      bar.className = 'bar';
      bar.style.height = Math.max(2, ((val - min) / range) * 36) + 'px';
      container.appendChild(bar);
    });
  }

  function appendMessageToDOM(role, content, isHtml) {
    var container = document.getElementById('chat-messages');
    var empty = document.getElementById('chat-empty');
    if (empty) empty.style.display = 'none';

    var div = document.createElement('div');
    div.className = 'chat-msg ' + role;
    var label = role === 'user' ? 'You' : 'Assistant';
    div.innerHTML = '<div class="msg-role">' + label + '</div><div class="msg-content"></div>';
    if (isHtml) {
      div.querySelector('.msg-content').innerHTML = content;
    } else {
      div.querySelector('.msg-content').textContent = content;
    }
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
      '<div style="color:var(--text-muted);margin:8px 0">Type a message below and press <kbd>Enter</kbd> to send</div>' +
      '<div style="color:var(--text-muted);font-size:0.8rem;margin-top:4px">Select a model from the dropdown above \u2022 Press <kbd>Shift+Enter</kbd> for new line</div>';
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
    if (tokens > 7000) { el.className = 'token-counter danger'; el.title = 'Warning: very long input, may exceed model context window'; }
    else if (tokens > 3000) { el.className = 'token-counter warn'; el.title = 'Getting long \u2014 some models may truncate'; }
    else { el.className = 'token-counter'; el.title = 'Estimated token count (4 chars \u2248 1 token)'; }
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
        // Nickname is non-critical — no banner
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
          tbody.innerHTML = '<tr><td colspan="4" class="text-muted" style="text-align:center;padding:24px">Leaderboard empty. No nodes have earned credits yet. Serve inference to earn credits.</td></tr>';
          return;
        }

        var html = '';
        for (var i = 0; i < entries.length; i++) {
          var e = entries[i];
          var tierClass = (e.tier || 'silver').toLowerCase().replace(/[^a-z]/g, '');
          html += '<tr>'
            + '<td class="mono">' + (e.rank || i+1) + '</td>'
            + '<td>' + (e.display_name !== e.node_id ? escapeHtml(e.display_name) + ' <span class="text-muted mono" style="font-size:0.75rem">' + escapeHtml(e.node_id) + '</span>' : '<span class="mono">' + escapeHtml(e.node_id) + '</span>') + '</td>'
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
        var code = escapeHtml(codes[i]);
        var d = escapeHtml(networkMap.paths[codes[i]] || '');
        svg += '<path id="region-' + code + '" d="' + d + '" fill="var(--bg-tertiary)" stroke="var(--border-light)" stroke-width="0.8" class="map-region" data-code="' + code + '"/>';
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
        // Network map is non-critical
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

      // Add pulsing dots at center of active regions (WI-16)
      var svg = document.querySelector('.world-svg');
      if (svg) {
        svg.querySelectorAll('.map-node-dot').forEach(function(d) { d.remove(); });
        var activeCodes = Object.keys(counts);
        for (var k = 0; k < activeCodes.length; k++) {
          var cc = activeCodes[k];
          var regionEl = document.getElementById('region-' + cc);
          if (!regionEl) continue;
          var bbox = regionEl.getBBox();
          var cx = bbox.x + bbox.width / 2;
          var cy = bbox.y + bbox.height / 2;
          var dotR = Math.max(3, Math.min(8, counts[cc] * 2));
          var dot = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
          dot.setAttribute('cx', cx);
          dot.setAttribute('cy', cy);
          dot.setAttribute('r', dotR);
          dot.setAttribute('fill', 'rgba(59,130,246,0.7)');
          dot.setAttribute('class', 'map-node-dot');
          svg.appendChild(dot);
        }
      }

      var statsEl = document.getElementById('map-stats-text');
      if (statsEl) statsEl.textContent = totalNodes + (totalNodes === 1 ? ' node' : ' nodes') + ' across ' + totalRegions + (totalRegions === 1 ? ' region' : ' regions');
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
      var statsEl = document.getElementById('map-stats-text');
      if (statsEl) statsEl.textContent = totalNodes + (totalNodes === 1 ? ' node' : ' nodes') + ' across ' + totalRegions + (totalRegions === 1 ? ' region' : ' regions');
      document.getElementById('map-legend-max').textContent = maxCount;
    },

    // ISO alpha-2 → country name (WI-16)
    countryNames: {US:'United States',CA:'Canada',MX:'Mexico',BR:'Brazil',AR:'Argentina',CL:'Chile',CO:'Colombia',GB:'United Kingdom',FR:'France',DE:'Germany',ES:'Spain',IT:'Italy',NL:'Netherlands',SE:'Sweden',NO:'Norway',FI:'Finland',PL:'Poland',UA:'Ukraine',RU:'Russia',TR:'Turkey',IN:'India',CN:'China',JP:'Japan',KR:'South Korea',AU:'Australia',NZ:'New Zealand',ZA:'South Africa',NG:'Nigeria',EG:'Egypt',KE:'Kenya',SG:'Singapore',ID:'Indonesia',TH:'Thailand',VN:'Vietnam',PH:'Philippines',TW:'Taiwan',IL:'Israel',AE:'UAE',SA:'Saudi Arabia',CH:'Switzerland',AT:'Austria',CZ:'Czech Republic',RO:'Romania',IE:'Ireland',PT:'Portugal',DK:'Denmark',BE:'Belgium'},

    showTooltip: function(event, code) {
      networkMap.hideTooltip();
      var info = networkMap.data && networkMap.data.regions ? networkMap.data.regions[code] : null;
      var tip = document.createElement('div');
      tip.id = 'map-tooltip';
      tip.className = 'map-tooltip';
      var countryName = networkMap.countryNames[code] || code;
      var html = '<strong>' + countryName + '</strong> <span class="text-muted" style="font-size:0.7rem">' + code + '</span>';
      if (info) {
        html += '<span class="mono" style="margin-left:8px">' + info.total + ' node' + (info.total !== 1 ? 's' : '') + '</span>';
        if (info.models) {
          var mids = Object.keys(info.models);
          if (mids.length > 0) {
            html += '<div class="mt-1" style="font-size:0.75rem">';
            for (var i = 0; i < Math.min(mids.length, 5); i++) {
              var mName = formatModelDisplayName(mids[i]);
              if (mName.length > 22) mName = mName.substring(0, 22) + '...';
              html += '<div class="flex-between" style="gap:12px"><span class="text-muted">' + escapeHtml(mName) + '</span><span class="mono">' + escapeHtml(String(info.models[mids[i]])) + '</span></div>';
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
    on('btn-skip-setup', 'click', function(e) {
      e.preventDefault();
      setup.finish();
    });
    // Setup provider select
    on('setup-provider-select', 'change', function() {
      var sel = document.getElementById('setup-provider-select');
      var inputDiv = document.getElementById('setup-provider-input');
      var signupLink = document.getElementById('setup-provider-signup');
      var providerUrls = {
        openai: 'https://platform.openai.com/api-keys',
        deepseek: 'https://platform.deepseek.com/api_keys',
        groq: 'https://console.groq.com/keys',
        nvidia_nim: 'https://build.nvidia.com/',
        cerebras: 'https://cloud.cerebras.ai/',
        sambanova: 'https://cloud.sambanova.ai/',
        anthropic: 'https://console.anthropic.com/settings/keys',
        mistral: 'https://console.mistral.ai/api-keys',
        fireworks: 'https://fireworks.ai/account/api-keys',
        together: 'https://api.together.xyz/settings/api-keys',
        deepinfra: 'https://deepinfra.com/dash/api_keys'
      };
      if (sel.value) {
        inputDiv.classList.remove('hidden');
        signupLink.href = providerUrls[sel.value] || '#';
      } else {
        inputDiv.classList.add('hidden');
      }
      document.getElementById('setup-provider-status').textContent = '';
    });
    on('setup-provider-save', 'click', async function() {
      var provider = document.getElementById('setup-provider-select').value;
      var key = document.getElementById('setup-provider-key').value.trim();
      var status = document.getElementById('setup-provider-status');
      if (!provider || !key) { status.textContent = 'Select a provider and enter a key'; status.style.color = 'var(--red)'; return; }
      status.textContent = 'Saving...'; status.style.color = 'var(--text-muted)';
      try {
        var body = {}; body[provider + '_key'] = key;
        var resp = await authFetch('/api/admin/providers', {method:'PUT', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body)});
        var data = await resp.json();
        if (data[provider]) {
          status.innerHTML = '<span style="color:var(--green)">✓ Connected!</span>';
          setup._savedProvider = provider;
        } else {
          status.innerHTML = '<span style="color:var(--red)">Key saved but provider not responding</span>';
          setup._savedProvider = provider;
        }
      } catch (e) { status.innerHTML = '<span style="color:var(--red)">Error: ' + e.message + '</span>'; }
    });
    // Wizard step indicators — clickable to jump to completed steps
    document.querySelectorAll('.wizard-step[data-step]').forEach(function(stepBtn) {
      stepBtn.addEventListener('click', function() {
        var target = parseInt(stepBtn.getAttribute('data-step'), 10);
        if (target < setup.currentStep) {
          setup.currentStep = target;
          setup.updateUI();
        }
      });
    });

    // Settings modal
    on('btn-close-settings', 'click', function() { ui.closeSettings(); });
    on('btn-copy-api-key', 'click', function() { settings.copyApiKey(); });
    on('btn-save-settings', 'click', function() { settings.save(); });
    on('btn-open-settings', 'click', function() { ui.openSettings(); });
    on('btn-rerun-setup', 'click', function() {
      localStorage.removeItem(SETUP_DONE_KEY);
      ui.closeSettings();
      setup.currentStep = 1;
      setup.updateUI();
      document.getElementById('setup-modal').classList.remove('hidden');
      setup.detectHardware();
    });

    // Provider test buttons (CSP-safe — data attribute binding)
    document.querySelectorAll('[data-test-provider]').forEach(function(btn) {
      btn.addEventListener('click', function() {
        settings.testProvider(btn.getAttribute('data-test-provider'));
      });
    });

    // Provider filter
    var providerFilter = document.getElementById('provider-filter');
    if (providerFilter) {
      providerFilter.addEventListener('input', function() {
        var q = this.value.toLowerCase();
        var cards = document.querySelectorAll('#provider-cards .provider-card');
        cards.forEach(function(card) {
          var name = (card.querySelector('strong') || {}).textContent || '';
          card.style.display = name.toLowerCase().indexOf(q) >= 0 ? '' : 'none';
        });
      });
    }

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
    on('chat-layout-toggle', 'click', function() { toggleChatLayout(); });

    // Image upload — file picker
    on('image-upload-btn', 'click', function() {
      document.getElementById('image-upload-input').click();
    });
    on('image-upload-input', 'change', function(e) {
      Array.from(e.target.files).forEach(addPendingImage);
      e.target.value = '';
    });

    // Image paste
    var chatInput = document.getElementById('chat-input');
    if (chatInput) {
      chatInput.addEventListener('paste', function(e) {
        var items = (e.clipboardData || {}).items || [];
        for (var i = 0; i < items.length; i++) {
          if (items[i].type.indexOf('image') !== -1) {
            e.preventDefault();
            addPendingImage(items[i].getAsFile());
          }
        }
      });
    }

    // Image drag-and-drop on chat area
    var chatArea = document.getElementById('view-chat');
    if (chatArea) {
      chatArea.addEventListener('dragover', function(e) {
        e.preventDefault();
        e.dataTransfer.dropEffect = 'copy';
      });
      chatArea.addEventListener('drop', function(e) {
        e.preventDefault();
        Array.from(e.dataTransfer.files).forEach(function(f) {
          if (f.type.startsWith('image/')) addPendingImage(f);
        });
      });
    }

    // Delegated buttons for dynamic CTA actions
    document.addEventListener('click', function(e) {
      var el = e.target.closest('[data-goto-chat],[data-goto-browse],[data-goto-settings],[data-goto-hf],[data-goto-network-code]') || e.target;
      if (el.getAttribute('data-goto-chat')) { ui.switchTab('chat'); }
      if (el.getAttribute('data-goto-browse')) { ui.openModelBrowser(); }
      if (el.getAttribute('data-goto-settings')) { ui.openSettings(true); }
      if (el.getAttribute('data-goto-hf')) { ui.openModelBrowser(); }
      if (el.getAttribute('data-goto-network-code')) { ui.switchTab('dashboard'); setTimeout(function() { var btn = document.getElementById('btn-share-network'); if (btn) btn.click(); }, 200); }
    });

    // Network discovery — share popover toggle
    on('btn-share-network', 'click', function(e) {
      e.stopPropagation();
      var pop = document.getElementById('share-popover');
      if (pop) pop.classList.toggle('show');
    });
    on('btn-copy-network-code', 'click', function() { copyNetworkCode(); });
    on('btn-join-network', 'click', function() { joinNetwork(); });

    // Network map
    on('map-model-filter', 'change', function() { networkMap.applyFilter(); });
    on('btn-refresh-map', 'click', function() { networkMap.refresh(); });

    // Model Compare
    on('btn-compare-run', 'click', function() { compare.run(); });

    // Leaderboard
    on('btn-refresh-leaderboard', 'click', function() { identity.loadLeaderboard(); });

    // Escape key closes open modals and shard context menu; Tab traps focus in modals
    document.addEventListener('keydown', function(e) {
      if (e.key === 'Escape') {
        shardMenu.hide();
        var settingsModal = document.getElementById('settings-modal');
        var modelModal = document.getElementById('model-browser-modal');
        if (settingsModal && !settingsModal.classList.contains('hidden')) { ui.closeSettings(); }
        else if (modelModal && !modelModal.classList.contains('hidden')) { ui.closeModelBrowser(); }
      }
      // Focus trap for open modals
      if (e.key === 'Tab') {
        var openModal = document.querySelector('.modal-overlay:not(.hidden) .modal');
        if (openModal) {
          var focusable = openModal.querySelectorAll('button, [href], input:not([type="hidden"]), select, textarea, [tabindex]:not([tabindex="-1"])');
          if (focusable.length > 0) {
            var first = focusable[0], last = focusable[focusable.length - 1];
            if (e.shiftKey && document.activeElement === first) { e.preventDefault(); last.focus(); }
            else if (!e.shiftKey && document.activeElement === last) { e.preventDefault(); first.focus(); }
          }
        }
      }
    });

    // Double-click to rename session titles in sidebar
    document.addEventListener('dblclick', function(e) {
      var target = e.target;
      var renameId = target.getAttribute('data-rename-session');
      if (renameId) {
        e.stopPropagation();
        e.preventDefault();
        chat.renameSession(renameId, target);
      }
    });

    // Delegated handlers for dynamically generated elements (CSP-safe)
    document.addEventListener('click', function(e) {
      var target = e.target;

      // Close share popover when clicking outside
      var pop = document.getElementById('share-popover');
      if (pop && pop.classList.contains('show') && !pop.contains(target) && target.id !== 'btn-share-network') {
        pop.classList.remove('show');
      }

      // Session delete button
      var delId = target.getAttribute('data-delete-session');
      if (delId) { e.stopPropagation(); chat.deleteSession(delId, e); return; }

      // Chat header title click to rename
      if (target.id === 'chat-header-title' && currentSessionId) {
        chat.renameSession(currentSessionId, target);
        return;
      }

      // Model action buttons
      var selectId = target.getAttribute('data-select-model');
      if (selectId) { selectModel(selectId); return; }

      var cloudId = target.getAttribute('data-select-cloud');
      if (cloudId) { selectModelDropdown(cloudId); ui.showBanner('success', 'Model selected: ' + cloudId); return; }

      var toggleTags = target.getAttribute('data-toggle-tags');
      if (toggleTags) {
        var hidden = document.getElementById(toggleTags);
        if (hidden) {
          var isHidden = hidden.style.display === 'none';
          hidden.style.display = isHidden ? 'inline' : 'none';
          target.textContent = isHidden ? 'Show less' : target.getAttribute('data-show-label') || 'Show all';
        }
        return;
      }

      var cancelId = target.getAttribute('data-cancel-download');
      if (cancelId) { cancelDownload(cancelId); return; }

      var requestId = target.getAttribute('data-request-model');
      if (requestId) { requestModel(requestId); return; }

      var removeId = target.getAttribute('data-remove-model');
      if (removeId) { removeModel(removeId); return; }

      // Unload model button (WI-3)
      var unloadId = target.getAttribute('data-unload-model');
      if (unloadId) { unloadModel(unloadId); return; }

      // HF download button
      var hfRepo = target.getAttribute('data-hf-download');
      if (hfRepo) { hf.download(hfRepo, target.getAttribute('data-hf-variant') || ''); return; }

      // Shard cell click → open context menu
      if (target.classList.contains('shard-cell')) {
        var shardModel = target.getAttribute('data-shard-model');
        var shardIdx = parseInt(target.getAttribute('data-shard-index'), 10);
        if (shardModel != null && !isNaN(shardIdx)) {
          var cls = target.className;
          var state = 'missing';
          if (cls.indexOf('local') !== -1) state = 'local';
          else if (cls.indexOf('downloading') !== -1 && cls.indexOf('peer-downloading') === -1) state = 'downloading';
          else if (cls.indexOf('peer') !== -1) state = 'peer';
          var isLocked = target.getAttribute('data-shard-locked') === '1';
          shardMenu.show(shardModel, shardIdx, state, e.clientX, e.clientY, isLocked);
          e.stopPropagation();
          return;
        }
      }

      // Shard context menu action button
      if (target.id === 'shard-ctx-action') { shardMenu.execute(); return; }

      // GGUF metadata toggle button
      var metaToggle = target.getAttribute('data-meta-toggle');
      if (metaToggle) { toggleMetadataPanel(metaToggle); return; }

      // Download queue cancel button
      var dlCancel = target.getAttribute('data-dl-cancel');
      if (dlCancel) { dlQueue.cancelDownload(dlCancel); return; }

      // Download queue log toggle
      var dlLogToggle = target.getAttribute('data-dl-log-toggle');
      if (dlLogToggle) {
        var logEl = document.querySelector('[data-dl-log="' + dlLogToggle + '"]');
        if (logEl) logEl.classList.toggle('open');
        return;
      }

      // Auto-manage gear icon
      var gearId = target.getAttribute('data-am-gear');
      if (gearId) { toggleAutoManagePanel(gearId); return; }

      // Auto-manage save button
      var amSave = target.getAttribute('data-am-save');
      if (amSave) { saveAutoManagePolicy(amSave); return; }

      // Chat retry button
      if (target.getAttribute('data-retry-chat')) {
        // Remove the error message and re-send the last user message
        var errMsg = target.closest('.chat-msg');
        if (errMsg) errMsg.remove();
        if (currentSessionId && sessions[currentSessionId]) {
          var msgs = sessions[currentSessionId].messages;
          // Pop the last user message, put it back in input, and re-send
          if (msgs.length > 0 && msgs[msgs.length - 1].role === 'user') {
            var lastUserMsg = msgs.pop();
            chat.saveSessions();
            document.getElementById('chat-input').value = lastUserMsg.content;
            chat.send();
          }
        }
        return;
      }

      // Close shard context menu on any other click
      shardMenu.hide();
    });
  }

  // ========================================================================
  // Collapsible Panels
  // ========================================================================
  function initCollapsiblePanels() {
    document.querySelectorAll('.panel-header[data-collapse]').forEach(function(header) {
      header.addEventListener('click', function() {
        var targetId = header.getAttribute('data-collapse');
        var body = document.getElementById(targetId);
        if (!body) return;
        body.classList.toggle('collapsed');
        header.classList.toggle('collapsed');
      });
    });
  }

  // ========================================================================
  // Mobile Model Selector Sync
  // ========================================================================
  function initMobileModelSync() {
    var mobile = document.getElementById('mobile-model-select');
    var mobileBtn = document.getElementById('btn-mobile-browse');

    // Sync mobile → desktop on change
    if (mobile) {
      mobile.addEventListener('change', function() {
        selectModelDropdown(mobile.value);
      });
    }

    // Mobile browse button opens model browser
    if (mobileBtn) {
      mobileBtn.addEventListener('click', function() {
        ui.openModelBrowser();
      });
    }
  }

  function syncMobileModelSelect() {
    var mobile = document.getElementById('mobile-model-select');
    if (!mobile) return;
    // Rebuild mobile select from dropdown data
    mobile.innerHTML = '';
    _modelDropdownData.forEach(function(m) {
      var opt = document.createElement('option');
      opt.value = m.id;
      opt.textContent = m.name;
      mobile.appendChild(opt);
    });
    mobile.value = currentModel;
  }

  // Format a raw model ID into a friendly display name
  function formatModelDisplayName(id) {
    if (!id) return 'Unknown';
    var name = id;
    // Strip common suffixes
    name = name.replace(/\.gguf$/i, '').replace(/-gguf$/i, '');
    // Remove repo prefix duplication: "tinyllama_tinyllama-1.1b" -> "tinyllama-1.1b"
    var parts = name.split(/[_]/);
    if (parts.length >= 2) {
      var prefix = parts[0].toLowerCase();
      var rest = parts.slice(1).join('_').toLowerCase();
      if (rest.indexOf(prefix) === 0) {
        name = parts.slice(1).join('_');
      }
    }
    // Preserve decimal numbers (1.1b, v0.3) by replacing dots between digits with placeholder
    name = name.replace(/(\d)\.(\d)/g, '$1\x00$2');
    // Split on separators and format each part
    return name.split(/[-_.]/).filter(Boolean).map(function(s) {
      s = s.replace(/\x00/g, '.'); // restore decimal dots
      // Keep quant tags uppercase (Q4_K_M, Q5_K_S, etc.)
      if (/^(q\d|iq\d|f16|f32|bf16)/i.test(s)) return s.toUpperCase();
      // Keep version strings as-is (v1, v0.3)
      if (/^v\d/i.test(s)) return s;
      // Keep size designators (1b, 7b, 1.1b)
      if (/^\d+\.?\d*[bBmM]$/.test(s)) return s.toUpperCase();
      return s.charAt(0).toUpperCase() + s.slice(1);
    }).join(' ');
  }

  // Enable/disable the chat panel based on model availability
  function updateChatAvailability(hasModels) {
    var sendBtn = document.getElementById('send-btn');
    var chatInput = document.getElementById('chat-input');
    var emptyState = document.querySelector('#chat-messages .chat-empty');

    if (sendBtn) sendBtn.disabled = !hasModels;
    if (chatInput) {
      chatInput.disabled = !hasModels;
      if (hasModels) {
        chatInput.placeholder = 'Type your message...';
      } else {
        // Check if a model is currently downloading (WI-9)
        var dlInfo = document.getElementById('chat-dl-progress');
        if (!dlInfo) chatInput.placeholder = 'No models available \u2014 download a model or share your Network Code to find peers';
      }
    }
    if (emptyState && !hasModels) {
      emptyState.innerHTML = '<div class="chat-empty-icon">&#11203;</div>' +
        '<div style="font-size:1.1rem;font-weight:600;color:var(--text-primary)">No Models Available</div>' +
        '<div style="color:var(--text-muted);margin:8px 0">Download models or share your Network Code to find peers — the swarm splits models across nodes so no single machine needs everything</div>' +
        '<div style="display:flex;gap:8px;margin-top:12px">' +
          '<button class="btn btn-primary" data-goto-browse="1">Download Model</button>' +
          '<button class="btn btn-outline" data-goto-network-code="1" style="border:1px solid var(--border)">Share Network Code</button>' +
        '</div>';
    }
  }

  // Show download progress inline above chat input when no model available (WI-9)
  function updateChatDownloadProgress(acquisitions) {
    var container = document.querySelector('.chat-input-area');
    if (!container) return;
    var existing = document.getElementById('chat-dl-progress');

    // Only show when no model is loaded
    var chatInput = document.getElementById('chat-input');
    if (chatInput && !chatInput.disabled) {
      if (existing) existing.remove();
      return;
    }

    if (!acquisitions || acquisitions.length === 0) {
      if (existing) existing.remove();
      return;
    }

    var active = acquisitions.find(function(a) {
      return typeof a.state === 'string' && (a.state === 'downloading' || a.state === 'awaiting_manifest');
    });
    if (!active) {
      if (existing) existing.remove();
      return;
    }

    var pct = 0;
    if (active.total_bytes > 0) pct = Math.min(100, Math.round((active.downloaded_bytes || 0) / active.total_bytes * 100));
    var speed = active.speed_bytes_per_sec || 0;
    var name = formatModelDisplayName(active.model_name || active.model_id || '');
    var text = 'Downloading ' + name + '... ' + pct + '%';
    if (speed > 0) text += ' (' + formatSpeed(speed) + ')';

    if (!existing) {
      existing = document.createElement('div');
      existing.id = 'chat-dl-progress';
      existing.className = 'chat-dl-progress';
      container.insertBefore(existing, container.firstChild);
    }
    existing.innerHTML = '<div class="chat-dl-bar"><div class="chat-dl-fill" style="width:' + pct + '%"></div></div>' +
      '<span class="chat-dl-text">' + escapeHtml(text) + '</span>';
  }

  // ========================================================================
  // Operation Mode Indicator
  // ========================================================================
  function updateModeIndicator(statsData, providerData) {
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

    var cloudProviders = [];
    if (providerData && providerData.providers) {
      providerData.providers.forEach(function(p) {
        if (p.configured) cloudProviders.push(p.name);
      });
    }

    // Build detail chips
    var chips = [];
    if (hasLocalModel) chips.push('<span class="mode-chip chip-local">' + hostedShards + ' shard' + (hostedShards !== 1 ? 's' : '') + ' local</span>');
    if (peers > 0) chips.push('<span class="mode-chip chip-peer">' + peers + ' peer' + (peers !== 1 ? 's' : '') + '</span>');
    var _providerNames = {
      openai: 'OpenAI', anthropic: 'Anthropic', deepseek: 'DeepSeek',
      mistral: 'Mistral', groq: 'Groq', nvidia_nim: 'NVIDIA NIM',
      cerebras: 'Cerebras', sambanova: 'SambaNova', fireworks: 'Fireworks',
      together: 'Together', deepinfra: 'DeepInfra', moonshot: 'Kimi'
    };
    cloudProviders.forEach(function(p) {
      chips.push('<span class="mode-chip chip-cloud">' + escapeHtml(_providerNames[p] || capitalize(p)) + '</span>');
    });

    // Remove old mode classes
    if (indicator) indicator.className = 'mode-indicator mb-2';

    var modeHelp = '';
    if (peers > 0 && hasLocalModel && cloudProviders.length > 0) {
      dot.className = 'mode-dot swarm';
      label.textContent = 'Swarm + Cloud';
      modeHelp = 'Full power — swarm inference with cloud fallback';
      if (indicator) indicator.classList.add('mode-hybrid');
    } else if (peers > 0 && hasLocalModel) {
      dot.className = 'mode-dot swarm';
      label.textContent = 'Swarm Mode';
      modeHelp = 'Running inference locally and with peers';
      if (indicator) indicator.classList.add('mode-swarm');
    } else if (peers > 0 && !hasLocalModel) {
      dot.className = 'mode-dot swarm';
      label.textContent = 'Swarm (remote)';
      modeHelp = 'Using peer nodes for inference (no local model)';
      if (indicator) indicator.classList.add('mode-swarm');
    } else if (hasLocalModel && cloudProviders.length > 0) {
      dot.className = 'mode-dot swarm';
      label.textContent = 'Local + Cloud';
      modeHelp = 'Local inference with cloud fallback — connect peers to go full swarm';
      if (indicator) indicator.classList.add('mode-hybrid');
    } else if (hasLocalModel) {
      dot.className = 'mode-dot offline';
      label.textContent = 'Solo Node';
      modeHelp = 'Local inference only — connect peers to unlock bigger models';
      if (indicator) indicator.classList.add('mode-offline');
      if (chips.length === 0) chips.push('<span class="mode-chip chip-none">Local only \u2014 share your Network Code to join the swarm</span>');
    } else if (cloudProviders.length > 0) {
      dot.className = 'mode-dot cloud';
      label.textContent = 'Cloud Only';
      modeHelp = 'Using cloud providers — download models or share your Network Code for free swarm inference';
      if (indicator) indicator.classList.add('mode-cloud');
    } else {
      dot.className = 'mode-dot offline';
      label.textContent = 'Ready to Join';
      modeHelp = 'Download models or share your Network Code to find peers';
      if (indicator) indicator.classList.add('mode-offline');
      chips = ['<span class="mode-chip chip-none" style="cursor:pointer" data-goto-hf="1">No models yet \u2014 <u>download models</u> or share your Network Code</span>'];
    }
    if (modeHelp) label.title = modeHelp;

    // Add quick-action button
    if (cloudProviders.length > 0 && !hasLocalModel && peers === 0) {
      chips.push('<button class="btn btn-sm" data-goto-hf="1" style="margin-left:8px;font-size:0.7rem;padding:2px 10px">Download Model</button>');
    }

    detail.innerHTML = chips.join(' ');
  }

  // Cache provider data so we can update mode indicator on stats updates
  var _cachedProviderData = null;

  async function loadModeIndicator() {
    var statsData = null;
    var providerData = null;
    try {
      var resp = await fetch('/api/admin/stats');
      if (resp.ok) statsData = await resp.json();
    } catch (e) {}
    try {
      var resp2 = await fetch('/api/admin/providers');
      if (resp2.ok) providerData = await resp2.json();
      _cachedProviderData = providerData;
    } catch (e) {}
    updateModeIndicator(statsData, providerData);
  }

  // ========================================================================
  // Chat Layout Toggle (Linear / Messenger)
  // ========================================================================
  function toggleChatLayout() {
    var container = document.getElementById('chat-messages');
    var btn = document.getElementById('chat-layout-toggle');
    var icon = document.getElementById('chat-layout-icon');
    var label = document.getElementById('chat-layout-label');
    if (!container) return;
    var isMessenger = container.classList.toggle('chat-messenger');
    if (icon) icon.innerHTML = isMessenger ? '&#9900;' : '&#9776;';
    if (label) label.textContent = isMessenger ? 'Messenger' : 'Linear';
    if (btn) btn.classList.toggle('active', isMessenger);
    try { localStorage.setItem(CHAT_LAYOUT_KEY, isMessenger ? 'messenger' : 'linear'); } catch(e) {}
    chat.scrollToBottom();
  }

  function initChatLayout() {
    try {
      var saved = localStorage.getItem(CHAT_LAYOUT_KEY);
      if (saved === 'messenger') {
        var container = document.getElementById('chat-messages');
        var icon = document.getElementById('chat-layout-icon');
        var label = document.getElementById('chat-layout-label');
        var btn = document.getElementById('chat-layout-toggle');
        if (container) container.classList.add('chat-messenger');
        if (icon) icon.innerHTML = '&#9900;';
        if (label) label.textContent = 'Messenger';
        if (btn) btn.classList.add('active');
      }
    } catch(e) {}
  }

  function init() {
    bindEvents();
    initCollapsiblePanels();
    initModelDropdown();
    initMobileModelSync();
    initChatLayout();

    inputEl = document.getElementById('chat-input');
    if (inputEl) {
      inputEl.addEventListener('input', autoResizeInput);
      inputEl.addEventListener('input', updateTokenCounter);
    }

    // Apply initial tab from URL (handles direct navigation / bookmarks)
    ui.switchTab(activeTab, true);

    // Handle browser back/forward navigation
    window.addEventListener('popstate', function(e) {
      var tab = (e.state && e.state.tab) ? e.state.tab : 'dashboard';
      ui.switchTab(tab, true);
    });

    chat.loadSessions();
    chat.renderSessionList();
    chat.renderMessages();

    setup.init();
    settings.init();
    settings.loadApiKey();
    dashboard.loadInitial();
    loadModels().then(function() { syncMobileModelSelect(); });
    loadPruneHistory();
    loadSchedule();
    loadModeIndicator();
    connectWebSocket();
    identity.loadNickname();

    // Start polling as fallback — will be paused once WebSocket connects
    startPolling();

    // Start provider health probe (default every 30s, configurable)
    startHealthPolling();
  }

  // Start when DOM is ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

  // Public API
  // ========================================================================
  // Model Compare Module — side-by-side multi-model comparison
  // ========================================================================
  var compare = {
    models: [],     // available models [{id, type}]
    selected: [],   // selected model IDs
    running: false,

    loadModels: async function() {
      try {
        var container = document.getElementById('compare-model-list');
        if (!container) return;

        // Fetch local + cloud models
        var localModels = [];
        var cloudModels = [];
        try {
          var resp = await authFetch('/api/admin/models');
          if (resp.ok) localModels = await resp.json();
        } catch(e) {}
        try {
          var resp2 = await authFetch('/api/admin/provider-models');
          if (resp2.ok) cloudModels = await resp2.json();
        } catch(e) {}

        compare.models = [];
        (localModels || []).forEach(function(m) {
          compare.models.push({ id: m.id || m.model_id || m.name, type: 'local' });
        });
        (cloudModels || []).forEach(function(m) {
          var mid = m.id || m.model_id || m.name;
          // Deduplicate
          if (!compare.models.some(function(x) { return x.id === mid; })) {
            compare.models.push({ id: mid, type: 'cloud' });
          }
        });

        if (compare.models.length === 0) {
          container.innerHTML = '<span class="text-muted" style="font-size:0.8rem">No models available. Download models, find peers via Network Code, or configure cloud providers in Settings.</span>';
          return;
        }

        container.innerHTML = '';
        compare.models.forEach(function(m) {
          var chip = document.createElement('label');
          chip.className = 'compare-model-chip';
          chip.innerHTML = '<input type="checkbox" value="' + escapeHtml(m.id) + '">' +
            '<span>' + escapeHtml(m.id) + '</span>' +
            '<span style="font-size:0.65rem;opacity:0.6">' + m.type + '</span>';
          chip.querySelector('input').addEventListener('change', function() {
            chip.classList.toggle('selected', this.checked);
            compare.updateSelected();
          });
          container.appendChild(chip);
        });
      } catch(e) {
        // non-critical
      }
    },

    updateSelected: function() {
      compare.selected = [];
      var checks = document.querySelectorAll('#compare-model-list input[type="checkbox"]:checked');
      checks.forEach(function(cb) { compare.selected.push(cb.value); });
    },

    run: async function() {
      if (compare.running) return;
      var prompt = (document.getElementById('compare-prompt') || {}).value;
      if (!prompt || !prompt.trim()) {
        showToast('Enter a prompt to compare', 'error');
        return;
      }
      if (compare.selected.length < 2) {
        showToast('Select at least 2 models to compare', 'error');
        return;
      }
      if (compare.selected.length > 10) {
        showToast('Maximum 10 models per comparison', 'error');
        return;
      }

      var system = (document.getElementById('compare-system') || {}).value || '';
      var temperature = parseFloat((document.getElementById('compare-temp') || {}).value) || 0.7;
      var maxTokens = parseInt((document.getElementById('compare-max-tokens') || {}).value) || 1024;

      compare.running = true;
      var btn = document.getElementById('btn-compare-run');
      if (btn) { btn.disabled = true; btn.textContent = 'Running...'; }

      var resultsDiv = document.getElementById('compare-results');
      var n = compare.selected.length;
      var colClass = n <= 2 ? 'cols-2' : n <= 3 ? 'cols-3' : n <= 4 ? 'cols-4' : 'cols-many';
      resultsDiv.className = 'compare-results ' + colClass;

      // Show spinner cards for each model
      resultsDiv.innerHTML = '';
      compare.selected.forEach(function(modelId) {
        var card = document.createElement('div');
        card.className = 'compare-card';
        card.id = 'compare-card-' + modelId.replace(/[^a-zA-Z0-9_-]/g, '_');
        card.innerHTML =
          '<div class="compare-card-header">' +
            '<span class="compare-card-model">' + escapeHtml(modelId) + '</span>' +
            '<span class="compare-card-meta"><span class="spinner" style="width:14px;height:14px"></span></span>' +
          '</div>' +
          '<div class="compare-card-body"><div class="compare-spinner"><div class="spinner"></div> Waiting for response...</div></div>';
        resultsDiv.appendChild(card);
      });

      var statusDiv = document.getElementById('compare-status');
      if (statusDiv) { statusDiv.style.display = ''; statusDiv.innerHTML = '<span class="text-muted">Sending prompt to ' + n + ' models concurrently...</span>'; }

      // Fire requests concurrently — use /v1/messages (Anthropic Messages API)
      var promises = compare.selected.map(function(modelId) {
        var body = {
          model: modelId,
          max_tokens: maxTokens,
          temperature: temperature,
          messages: [{ role: 'user', content: prompt.trim() }],
          stream: false,
        };
        if (system.trim()) body.system = system.trim();

        var start = performance.now();
        return authFetch('/v1/messages', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body),
        }).then(function(resp) {
          var elapsed = Math.round(performance.now() - start);
          return resp.json().then(function(data) {
            return { model: modelId, data: data, ok: resp.ok, latency_ms: elapsed };
          });
        }).catch(function(err) {
          return { model: modelId, error: err.message, ok: false, latency_ms: Math.round(performance.now() - start) };
        });
      });

      // Update cards as results come in
      var completed = 0;
      promises.forEach(function(p) {
        p.then(function(result) {
          completed++;
          compare.renderCard(result);
          if (statusDiv) {
            statusDiv.innerHTML = '<span class="text-muted">' + completed + ' / ' + n + ' models complete</span>';
            if (completed === n) {
              statusDiv.innerHTML = '<span style="color:var(--green)">All ' + n + ' models complete</span>';
              setTimeout(function() { statusDiv.style.display = 'none'; }, 3000);
            }
          }
        });
      });

      // Wait for all to finish
      Promise.all(promises).then(function() {
        compare.running = false;
        if (btn) { btn.disabled = false; btn.textContent = 'Run Compare'; }
      });
    },

    renderCard: function(result) {
      var cardId = 'compare-card-' + result.model.replace(/[^a-zA-Z0-9_-]/g, '_');
      var card = document.getElementById(cardId);
      if (!card) return;

      var content = '';
      var isError = false;
      var inputTokens = 0;
      var outputTokens = 0;

      if (result.error) {
        content = result.error;
        isError = true;
      } else if (!result.ok) {
        content = result.data.error && result.data.error.message
          ? result.data.error.message
          : JSON.stringify(result.data.error || result.data, null, 2);
        isError = true;
      } else {
        // Anthropic Messages API response
        var blocks = result.data.content || [];
        blocks.forEach(function(b) {
          if (b.type === 'text' && b.text) content += b.text;
        });
        if (!content) content = '(empty response)';
        inputTokens = (result.data.usage || {}).input_tokens || 0;
        outputTokens = (result.data.usage || {}).output_tokens || 0;
      }

      card.innerHTML =
        '<div class="compare-card-header">' +
          '<span class="compare-card-model">' + escapeHtml(result.model) + '</span>' +
          '<span class="compare-card-meta">' +
            '<span>' + result.latency_ms + 'ms</span>' +
            (isError ? '<span style="color:var(--red,#ff6464)">error</span>' : '<span style="color:var(--green)">ok</span>') +
          '</span>' +
        '</div>' +
        '<div class="compare-card-body' + (isError ? ' error' : '') + '">' + escapeHtml(content) + '</div>' +
        (isError ? '' :
          '<div class="compare-card-footer">' +
            '<span>In: ' + inputTokens + ' tokens</span>' +
            '<span>Out: ' + outputTokens + ' tokens</span>' +
            '<span>Latency: ' + result.latency_ms + 'ms</span>' +
          '</div>'
        );
    },
  };

  // --- Network invite code ---
  async function loadNetworkCode() {
    try {
      var resp = await authFetch('/api/admin/network-code');
      var data = await resp.json();
      var panel = document.getElementById('invite-code-panel');
      if (!panel) return;

      var phase = data.phase || 'seedling';
      var peerCount = data.peer_count || 0;
      var badge = document.getElementById('network-phase-badge');
      if (badge) {
        if (phase === 'established') {
          badge.textContent = peerCount + ' peer' + (peerCount !== 1 ? 's' : '') + ' connected';
          badge.className = 'badge badge-green';
        } else {
          badge.textContent = 'No peers';
          badge.className = 'badge badge-orange';
        }
      }

      // Always show the panel — users need it to share/join even when connected
      panel.style.display = '';
      var codeInput = document.getElementById('my-network-code');
      if (codeInput && data.code) codeInput.value = data.code;
    } catch (e) {
      // Network code is non-critical on startup — no banner needed
    }
  }

  function copyNetworkCode() {
    var input = document.getElementById('my-network-code');
    var btn = document.getElementById('btn-copy-network-code');
    if (input && input.value) {
      navigator.clipboard.writeText(input.value).then(function() {
        if (btn) { btn.textContent = 'Copied!'; btn.style.color = 'var(--green)'; setTimeout(function() { btn.textContent = 'Copy'; btn.style.color = ''; }, 2000); }
        showToast('Network code copied to clipboard', 'success');
      }).catch(function() {
        ui.showBanner('error', 'Failed to copy \u2014 try selecting and copying manually');
      });
    }
  }

  async function joinNetwork() {
    var input = document.getElementById('join-code-input');
    var status = document.getElementById('join-status');
    if (!input || !input.value.trim()) return;

    if (status) { status.textContent = 'Connecting...'; status.style.color = 'var(--text-muted)'; }

    try {
      var resp = await authFetch('/api/admin/join-network', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ code: input.value.trim() })
      });
      var data = await resp.json();
      if (resp.ok) {
        if (status) { status.textContent = 'Connected! Peer added.'; status.style.color = 'var(--green)'; }
        input.value = '';
        showToast('Peer connected successfully', 'success');
        // Refresh dashboard data after a short delay to show the new peer
        setTimeout(function() { loadNetworkCode(); }, 2000);
      } else {
        if (status) { status.textContent = data.error || 'Failed to join'; status.style.color = 'var(--red, #ff6464)'; }
      }
    } catch (e) {
      if (status) { status.textContent = 'Network error'; status.style.color = 'var(--red, #ff6464)'; }
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
    compare: compare,
    requestModel: requestModel,
    selectModel: selectModel,
    cancelDownload: cancelDownload,
    removeModel: removeModel,
    shutdown: shutdown,
    copyNetworkCode: copyNetworkCode,
    joinNetwork: joinNetwork,
    openModelBrowser: function() { ui.openModelBrowser(); },
    switchTab: function(t) { ui.switchTab(t); },
  };
})();
