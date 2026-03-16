'use strict';

// ============================================================================
// SwarmLLM — Unified single-page application
// ============================================================================

// ============================================================================
// Provider icons — bundled SVGs served from /static/icons/
// ============================================================================
var _ICON_BASE = '/static/icons/';
var _ICON_MAP = {
  openai:     'openai',
  anthropic:  'anthropic',
  deepseek:   'deepseek-color',
  mistral:    'mistral-color',
  groq:       'groq',
  nvidia_nim: 'nvidia-color',
  cerebras:   'cerebras-color',
  sambanova:  'sambanova-color',
  fireworks:  'fireworks-color',
  together:   'together-color',
  deepinfra:  'deepinfra-color',
  moonshot:   'moonshot',
  // model-family → icon (for local/swarm models)
  llama:      'meta-color',
  gemma:      'gemma-color',
  gemini:     'gemini-color',
  qwen:       'qwen-color',
  phi:        'microsoft-color',
  claude:     'claude-color',
};

function providerIconUrl(key) {
  var id = _ICON_MAP[key];
  return id ? _ICON_BASE + id + '.svg' : null;
}

// Returns an <img> tag string for a provider/model icon, or '' if unknown.
// size defaults to 16.
function providerIconHtml(key, size) {
  var url = providerIconUrl(key);
  if (!url) return '';
  size = size || 16;
  return '<img src="' + url + '" width="' + size + '" height="' + size + '" alt="" aria-hidden="true" class="provider-icon" style="display:inline-block;vertical-align:middle;flex-shrink:0">';
}

// Infer a provider/family key from a model ID string.
function modelIconKey(modelId) {
  if (!modelId) return null;
  var m = modelId.toLowerCase();
  if (m.startsWith('gpt') || m.startsWith('o1') || m.startsWith('o3') || m.startsWith('o4') || m.includes('-openai')) return 'openai';
  if (m.startsWith('claude')) return 'claude';
  if (m.startsWith('deepseek')) return 'deepseek';
  if (m.startsWith('mistral') || m.startsWith('mixtral') || m.startsWith('codestral')) return 'mistral';
  if (m.startsWith('llama') || m.startsWith('meta-llama')) return 'llama';
  if (m.startsWith('gemma')) return 'gemma';
  if (m.startsWith('gemini')) return 'gemini';
  if (m.startsWith('qwen')) return 'qwen';
  if (m.startsWith('phi')) return 'phi';
  return null;
}

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
  try {
    var _cached = sessionStorage.getItem('swarmllm_model_status');
    if (_cached) modelStatus = JSON.parse(_cached);
  } catch (e) {}
  var _modelStatusPending = {}; // track in-flight probes


  // --- THEME (light / dark / system) ---
  var THEME_KEY = 'swarmllm_theme';

  function applyTheme(theme) {
    var resolved = theme;
    if (theme === 'system') {
      resolved = window.matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark';
    }
    document.documentElement.setAttribute('data-theme', resolved);
    var btn = document.getElementById('btn-theme-toggle');
    var icons = { dark: '\u263E', light: '\u2600', system: '\u25D1' };
    if (btn) btn.textContent = icons[theme] || '\u263E';
  }

  // Listen for system theme changes when in 'system' mode
  try {
    window.matchMedia('(prefers-color-scheme: light)').addEventListener('change', function() {
      if ((localStorage.getItem(THEME_KEY) || 'dark') === 'system') applyTheme('system');
    });
  } catch(e) {}

  // --- HELPERS ---
  function escapeHtml(str) {
    var div = document.createElement('div');
    div.textContent = str || '';
    return div.innerHTML;
  }

  // Authenticated fetch — adds Bearer token to all requests that need auth
  async function authFetch(url, opts) {
    // Ensure API key is loaded before making authenticated requests
    if (!settings._apiKeyFull && settings._apiKeyPromise) {
      await settings._apiKeyPromise;
    }
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
      // Show/hide sidebar based on tab
      var sidebar = document.getElementById('sidebar');
      var edgeTrigger = document.getElementById('sidebar-edge-trigger');
      if (sidebar) {
        if (tab === 'chat') {
          sidebar.style.display = '';
          sidebar.classList.remove('sidebar-float');
          // Auto-open on desktop, keep collapsed on mobile (user uses hamburger)
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
        chat.scrollToBottom();
        document.getElementById('chat-input').focus();
      }
      if (tab === 'leaderboard') {
        identity.loadLeaderboard();
      }
      if (tab === 'network-map') {
        networkMap.refresh();
      }
      if (tab === 'compare' && typeof compare !== 'undefined' && compare) {
        compare.loadModels();
        compare.renderHistory();
      }
    },

    openSidebar: function() {
      var sidebar = document.getElementById('sidebar');
      var overlay = document.getElementById('sidebar-overlay');
      if (sidebar) sidebar.classList.remove('collapsed');
      // Show overlay only on mobile
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
        ui.closeSidebar();
      } else {
        ui.openSidebar();
      }
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
      removeBtn.style.cssText = 'position:absolute;top:-4px;right:-4px;background:var(--red);color:#fff;border:none;border-radius:50%;width:18px;height:18px;font-size:12px;cursor:pointer;line-height:18px;padding:0;';
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
      // Clean up any stale empty sessions before creating a new one
      var emptied = [];
      Object.keys(sessions).forEach(function(sid) {
        if (sessions[sid].messages.length === 0 && sid !== currentSessionId) {
          emptied.push(sid);
          delete sessions[sid];
        }
      });
      var id = 'session_' + Date.now();
      sessions[id] = { id: id, title: 'New Chat', messages: [], created: Date.now(), model: currentModel || '' };
      currentSessionId = id;
      chat.saveSessions();
      chat.renderSessionList();
      chat.renderMessages();
      chat.updateChatHeader();
      if (emptied.length > 0) {
        showToast('Cleaned up ' + emptied.length + ' empty session' + (emptied.length > 1 ? 's' : ''), 'info', 3000);
      }
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

      // Close sidebar after selecting a chat on mobile (overlay mode)
      if (window.innerWidth < 768) ui.closeSidebar();
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
        div.onclick = function() { chat.switchSession(s.id); if (activeTab !== 'chat') ui.switchTab('chat'); };
        var title = s.title.length > 28 ? s.title.substring(0, 28) + '...' : s.title;
        var timeStr = '';
        if (s.created) {
          var d = new Date(s.created);
          timeStr = d.toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' });
        }
        var modelItem = s.model ? _modelDropdownData.find(function(m) { return m.id === s.model; }) : null;
        var source = getModelSource(s.model || '');
        var sourceIcon = source === 'local' ? '&#128187;' : source === 'cloud' ? '&#9729;' : '&#11042;';
        var sourceLabel = source === 'local' ? 'Your PC' : source === 'cloud' ? 'Cloud' : 'Swarm';
        var isEncrypted = modelItem && modelItem.encrypted;
        var encIcon = isEncrypted ? ' &#128274;' : '';
        var badgeClass = 'session-model-badge session-source-' + source;
        var tooltipParts = [escapeHtml(s.model || '')];
        if (source !== 'local') tooltipParts.push(sourceLabel);
        if (isEncrypted) tooltipParts.push('Encrypted pipeline (end-to-end)');
        var modelBadge = s.model ? '<span class="' + badgeClass + '" title="' + tooltipParts.join(' \u2022 ') + '">' + sourceIcon + ' ' + escapeHtml(formatModelDisplayName(s.model)) + encIcon + '</span>' : '';
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
      var headerModelItem = s.model ? _modelDropdownData.find(function(m) { return m.id === s.model; }) : null;
      var headerEncIcon = (headerModelItem && headerModelItem.encrypted) ? ' <span class="badge-encrypted" title="Encrypted pipeline active">&#128274;</span>' : '';
      var msgCount = s.messages.length;
      var countLabel = msgCount === 0 ? 'New' : (msgCount === 1 ? '1 message' : msgCount + ' messages');
      var countClass = 'chat-session-count' + (msgCount === 0 ? ' is-new' : '');
      header.classList.add('visible');
      header.innerHTML =
        '<span class="chat-session-title" id="chat-header-title" title="Click to rename">' + escapeHtml(s.title) + '</span>' +
        '<span class="' + countClass + '">' + escapeHtml(countLabel) + '</span>' +
        '<span class="' + badgeClass + '" title="' + escapeHtml(badgeTitle) + '">' + escapeHtml(modelName) + (available ? '' : ' (unavailable)') + headerEncIcon + '</span>';
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
            html += '<img src="' + escapeHtml(url) + '" style="max-height:120px;max-width:200px;border-radius:8px;margin-right:4px;" />';
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
        ui.showBanner('warning', 'No model selected \u2014 pick one from the dropdown above, or click + to download one');
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
          userHtml += '<img src="' + escapeHtml(img.data_url) + '" style="max-height:120px;max-width:200px;border-radius:8px;margin-right:4px;" />';
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
          contentEl.textContent = 'No response received. The model might still be loading \u2014 wait a moment and try again.';
          contentEl.classList.add('chat-error');
        }
      } catch (e) {
        if (!fullContent) {
          contentEl.textContent = 'Connection failed \u2014 check that the server is running and try again.';
          contentEl.classList.add('chat-error');
        }
      }

      // Show response time — append inside bubble so it renders correctly in both layouts
      var elapsed = ((performance.now() - startTime) / 1000).toFixed(2);
      var timerEl = document.createElement('div');
      timerEl.className = 'msg-timer';
      timerEl.textContent = 'Response time: ' + elapsed + 's';
      var timerTarget = assistantEl.querySelector('.msg-bubble') || assistantEl;
      timerTarget.appendChild(timerEl);

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
        // Strip image data URLs before persisting — they bloat localStorage and
        // persist potentially personal images indefinitely
        var stripped = {};
        Object.keys(sessions).forEach(function(id) {
          var s = sessions[id];
          stripped[id] = Object.assign({}, s, {
            messages: (s.messages || []).map(function(m) {
              if (m.images && m.images.length > 0) {
                var copy = Object.assign({}, m);
                delete copy.images;
                return copy;
              }
              return m;
            })
          });
        });
        localStorage.setItem(SESSIONS_KEY, JSON.stringify(stripped));
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
        ui.showBanner('error', "Can't reach SwarmLLM — is it running?");
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
        // Ensure API key is loaded before fetching authenticated endpoints
        if (settings._apiKeyPromise) await settings._apiKeyPromise;

        var resp = await fetch('/api/admin/models');
        var models = await resp.json();
        var cloudModels = [];
        try {
          var pmResp = await authFetch('/api/admin/provider-models');
          if (pmResp.ok) {
            var pmData = await pmResp.json();
            cloudModels = pmData.models || [];
          }
        } catch (e) {}
        dashboard.renderModels(models, cloudModels);
        // If cloud models came back empty, retry once after 3s (provider APIs may be slow on cold start)
        if (cloudModels.length === 0) {
          setTimeout(async function() {
            try {
              var retry = await authFetch('/api/admin/provider-models');
              if (retry.ok) {
                var rd = await retry.json();
                if (rd.models && rd.models.length > 0) {
                  dashboard.renderModels(models, rd.models);
                }
              }
            } catch(e) {}
          }, 3000);
        }
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
          if (hw.gpu_vram_mb) {
            var vramUsed = hw.gpu_vram_used_mb || 0;
            document.getElementById('node-vram').textContent = formatMB(vramUsed) + ' / ' + formatMB(hw.gpu_vram_mb) + ' VRAM';
            var vramPct = hw.gpu_vram_mb > 0 ? (vramUsed / hw.gpu_vram_mb * 100) : 0;
            document.getElementById('vram-bar').style.width = vramPct.toFixed(1) + '%';
            document.getElementById('vram-bar').className = vramPct > 90 ? 'fill red' : (vramPct > 70 ? 'fill orange' : 'fill cyan');
          }
        } else {
          document.getElementById('node-gpu').textContent = 'CPU only';
          document.getElementById('node-vram').textContent = '';
          document.getElementById('vram-bar').style.width = '0%';
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

      // Feed neural background
      if (typeof NeuralBg !== 'undefined') NeuralBg.updateState(data);
    },

    renderModels: function(models, cloudModels) {
      window._lastModelsData = models || [];
      var list = document.getElementById('models-list');
      var empty = document.getElementById('models-empty');
      var loading = document.getElementById('models-loading');
      if (loading) loading.remove();

      var hasCloud = cloudModels && cloudModels.length > 0;
      if ((!models || models.length === 0) && !hasCloud) {
        list.innerHTML = '';
        empty.style.display = '';
        var _sb = document.getElementById('models-stats-bar');
        if (_sb) _sb.style.display = 'none';
        return;
      }

      // Filter out ghost models: no local shards, no peer holders, not downloading
      models = models.filter(function(m) {
        if (m.local || m.hosted_shards > 0) return true;
        if (m.peers_hosting > 0) return true;
        if (m.acquisition === 'downloading') return true;
        var anyHolder = (m.shards || []).some(function(s) { return s.holders > 0; });
        return anyHolder;
      });

      if (models.length === 0 && !hasCloud) {
        list.innerHTML = '';
        empty.style.display = '';
        var _sb2 = document.getElementById('models-stats-bar');
        if (_sb2) _sb2.style.display = 'none';
        return;
      }

      empty.style.display = 'none';
      list.innerHTML = '';

      // ── Quick stats ─────────────────────────────────────────────────────────
      var statsBar = document.getElementById('models-stats-bar');
      if (statsBar) {
        var statLocal = models.filter(function(m) { return m.local || m.hosted_shards > 0; }).length;
        var statReady = models.filter(function(m) {
          var hc = m.hosted_shards || 0, sc = m.shard_count || (m.shards || []).length;
          return m.status === 'loaded' || m.status === 'ready' || (hc === sc && sc > 0);
        }).length;
        var statNet = models.filter(function(m) { return !m.local && !(m.hosted_shards > 0) && m.peers_hosting > 0; }).length;
        var statCloudTotal = hasCloud ? cloudModels.length : 0;
        var statProviders = 0;
        if (hasCloud) {
          var _pset = {};
          cloudModels.forEach(function(cm) { _pset[cm.provider || 'cloud'] = 1; });
          statProviders = Object.keys(_pset).length;
        }
        document.getElementById('stat-chip-ready-val').textContent = statReady;
        document.getElementById('stat-chip-network-val').textContent = statNet;
        document.getElementById('stat-chip-cloud-val').textContent = statCloudTotal;
        document.getElementById('stat-chip-providers-val').textContent = statProviders;
        statsBar.style.display = '';
        // Remote chip: only show when > 0
        var netChip = document.getElementById('stat-chip-network');
        if (netChip) netChip.style.display = statNet > 0 ? '' : 'none';
        // Cloud group + separator: only show when cloud providers connected
        var cloudGroup = document.getElementById('stat-group-cloud');
        var sep = statsBar.querySelector('.models-stat-sep');
        if (cloudGroup) cloudGroup.style.display = hasCloud ? '' : 'none';
        if (sep) sep.style.display = hasCloud ? '' : 'none';
      }

      // ── Swarm models section ─────────────────────────────────────────────────
      var swarmBody;
      if (models.length > 0) {
        var swarmSection = document.createElement('details');
        swarmSection.className = 'models-section';
        swarmSection.open = true;
        var swarmReadyCount = models.filter(function(m) {
          var hc = m.hosted_shards || 0, sc = m.shard_count || (m.shards || []).length;
          return m.status === 'loaded' || m.status === 'ready' || (hc === sc && sc > 0);
        }).length;
        var swarmMeta = models.length + ' model' + (models.length !== 1 ? 's' : '') +
          (swarmReadyCount > 0 ? ' \u00b7 ' + swarmReadyCount + ' ready' : '');
        swarmSection.innerHTML = '<summary class="models-section-header">' +
          '<img src="/static/icons/swarm.svg" width="16" height="16" alt="" aria-hidden="true" class="models-section-logo">' +
          '<span class="models-section-title">Swarm Models</span>' +
          '<span class="models-section-count">' + swarmMeta + '</span>' +
          '</summary>';
        swarmBody = document.createElement('div');
        swarmBody.className = 'models-section-body';
        swarmSection.appendChild(swarmBody);
        list.appendChild(swarmSection);
      }

      models.forEach(function(m) {
        var shards = m.shards || [];
        // Derive shard count from actual shard list if API field missing/zero
        var shardCount = m.shard_count || shards.length || 0;
        var hostedShards = m.hosted_shards || 0;
        var globalAvail = m.global_available || hostedShards;
        var isDownloading = m.acquisition === 'downloading';
        var isReady = m.status === 'loaded' || m.status === 'ready' || (globalAvail === shardCount && shardCount > 0);
        var isPartial = !isReady && hostedShards > 0 && hostedShards < shardCount;
        var safeId = (m.id || '').replace(/[^a-zA-Z0-9]/g, '_');

        var card = document.createElement('div');
        card.className = 'model-card' + (isReady ? ' ready' : (isDownloading ? ' downloading' : (isPartial ? ' partial' : '')));
        card.setAttribute('data-model-id', m.id);

        // Status pill (compact, for title bar)
        var statusHtml = '';
        if (m.status === 'loaded') {
          statusHtml = '<span class="model-status-pill active">● Active</span>';
        } else if (isReady) {
          statusHtml = '<span class="model-status-pill ready">Ready</span>';
        } else if (isDownloading) {
          statusHtml = '<span class="model-status-pill downloading"><span class="spinner" style="width:10px;height:10px;border-width:1.5px;vertical-align:middle;margin-right:3px"></span>Downloading</span>';
        } else if (isPartial) {
          statusHtml = '<span class="model-status-pill partial">' + hostedShards + '/' + shardCount + ' local</span>';
        } else {
          statusHtml = '<span class="model-status-pill network">On network</span>';
        }

        // Trust level badge
        var trustBadge = '';
        if (m.trust_level === 'network_popular') {
          trustBadge = '<span class="badge-trust badge-trust-popular" title="Widely hosted across the network">Popular</span>';
        } else if (m.trust_level === 'demand_verified') {
          trustBadge = '<span class="badge-trust badge-trust-verified" title="Has received real inference requests">Verified</span>';
        } else if (m.trust_level === 'pinned') {
          trustBadge = '<span class="badge-trust badge-trust-pinned" title="Manually approved by you">Pinned</span>';
        }

        // Encrypted pipeline badge
        var encBadge = '';
        if (m.shard_count > 1 && m.local) {
          var encReady = m.has_first_shard && m.has_last_shard;
          var encActive = m.encrypted_pipeline;
          var encClass = encActive ? 'badge-encrypted active' : (encReady ? 'badge-encrypted ready' : 'badge-encrypted faded');
          var encTitle = encActive ? 'Encrypted pipeline active \u2014 click to disable' :
            (encReady ? 'Encrypted pipeline available \u2014 click to enable' :
              'Encrypted pipeline unavailable \u2014 need first + last shard');
          var missingParts = [];
          if (!m.has_first_shard) missingParts.push('first (shard 0)');
          if (!m.has_last_shard) missingParts.push('last (shard ' + (m.shard_count - 1) + ')');
          if (missingParts.length > 0) encTitle += '. Missing: ' + missingParts.join(', ');
          encBadge = '<span class="' + encClass + '" data-enc-toggle="' + escapeHtml(m.id) + '" data-enc-ready="' + (encReady ? '1' : '0') + '" title="' + escapeHtml(encTitle) + '">&#128274;</span>';
        }

        // Source label
        var sourceLabel = '';
        if (m.source === 'network' && hostedShards === 0) {
          sourceLabel = '<span class="badge badge-remote" title="Available via network peers">Remote</span>';
        }

        // Gear + info buttons
        var gearHtml = '<button class="model-gear-btn" data-am-gear="' + escapeHtml(m.id) + '" title="Auto-manage settings">&#9881;</button>';
        var metaBtnHtml = m.has_header ? '<button class="model-meta-btn" data-meta-toggle="' + escapeHtml(m.id) + '" title="GGUF Metadata">&#9432;</button>' : '';

        // ── Shard grid (always numbered, 3 size tiers) ──────────────────────
        var shardHtml = '';
        if (shards.length > 0) {
          var lastIdx = shardCount - 1;
          // Size tier: adapt cell size to shard count
          var sizeClass = shardCount > 50 ? ' shard-grid-sm' : (shardCount > 20 ? ' shard-grid-md' : '');
          shardHtml = '<div class="shard-grid' + sizeClass + '" data-model-grid="' + safeId + '">';
          var localCount = 0, peerCount = 0, dlCount = 0, peerDlCount = 0, queuedCount = 0, missingCount = 0;

          shards.forEach(function(s) {
            var cls = 'missing';
            var label = '' + s.index;
            var dlPct = 0;

            if (s.local) { cls = 'local'; localCount++; }
            else if (s.holders > 0) { cls = 'peer'; peerCount++; }
            else { missingCount++; }

            if (s.download && s.download.state === 'Downloading') {
              dlPct = s.download.progress_pct || 0;
              cls = 'downloading'; dlCount++;
              label = dlPct + '%';
              if (missingCount > 0) missingCount--;
              if (peerCount > 0 && !s.local) peerCount--;
            } else if (s.download && s.download.state === 'Verifying') {
              cls = 'verifying'; dlCount++;
              label = '\u2713';
              if (missingCount > 0) missingCount--;
              if (peerCount > 0 && !s.local) peerCount--;
            }

            if (s.peer_downloads && s.peer_downloads.length > 0) {
              if (cls !== 'local' && cls !== 'downloading' && cls !== 'verifying') {
                dlPct = s.peer_downloads[0].progress_pct || 0;
                cls = 'peer-downloading'; peerDlCount++;
                label = dlPct + '%';
                if (missingCount > 0) missingCount--;
                if (peerCount > 0) peerCount--;
              }
            }

            var title = 'Shard ' + s.index + (s.size_bytes ? ' (' + formatBytes(s.size_bytes) + ')' : '');
            if (cls === 'local') title += ' \u2014 Stored locally';
            else if (cls === 'peer') title += ' \u2014 Available from ' + s.holders + ' peer' + (s.holders !== 1 ? 's' : '');
            else if (cls === 'downloading') title += ' \u2014 Downloading (' + dlPct + '%)';
            else if (cls === 'verifying') title += ' \u2014 Verifying (BLAKE3)';
            else if (cls === 'peer-downloading') title += ' \u2014 Peer downloading (' + dlPct + '%)';
            else title += ' \u2014 Not available';

            var style = '';
            if (cls === 'downloading' || cls === 'peer-downloading') {
              style = ' style="--dl-pct:' + dlPct + '%"';
            }

            var lockIcon = s.locked ? '<span class="shard-lock-icon" title="Locked (pinned)">\uD83D\uDD12</span>' : '';

            // First/last shard: always mark as endpoints (pipeline structural boundaries)
            var endpointClass = '';
            if (shardCount > 1 && (s.index === 0 || s.index === lastIdx)) {
              if (m.encrypted_pipeline && s.local) {
                endpointClass = ' shard-pinned';
              } else {
                endpointClass = ' shard-endpoint';
              }
            }

            shardHtml += '<div class="shard-cell ' + cls + (s.locked ? ' locked' : '') + endpointClass + '"' + style +
              ' data-shard="' + safeId + '-' + s.index + '"' +
              ' data-shard-model="' + escapeHtml(m.id) + '"' +
              ' data-shard-index="' + s.index + '"' +
              ' data-shard-locked="' + (s.locked ? '1' : '0') + '"' +
              ' title="' + escapeHtml(title) + '">' + label + lockIcon + '</div>';
          });
          shardHtml += '</div>';

          // Shard summary counts
          var summaryParts = [];
          if (localCount > 0) summaryParts.push('<span class="shard-sum-item shard-sum-local"><span class="shard-sum-dot"></span>' + localCount + ' local</span>');
          if (peerCount > 0) summaryParts.push('<span class="shard-sum-item shard-sum-peer"><span class="shard-sum-dot"></span>' + peerCount + ' peer' + (peerCount !== 1 ? 's' : '') + '</span>');
          if (dlCount > 0) summaryParts.push('<span class="shard-sum-item shard-sum-dl"><span class="shard-sum-dot"></span>' + dlCount + ' downloading</span>');
          if (peerDlCount > 0) summaryParts.push('<span class="shard-sum-item shard-sum-peer-dl"><span class="shard-sum-dot"></span>' + peerDlCount + ' peer DL</span>');
          if (queuedCount > 0) summaryParts.push('<span class="shard-sum-item shard-sum-queued"><span class="shard-sum-dot"></span>' + queuedCount + ' queued</span>');
          if (missingCount > 0) summaryParts.push('<span class="shard-sum-item shard-sum-missing"><span class="shard-sum-dot"></span>' + missingCount + ' missing</span>');
          if (summaryParts.length > 0) {
            shardHtml += '<div class="shard-summary" data-model-summary="' + safeId + '">' + summaryParts.join('') + '</div>';
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
            etaStr = formatEta((totalBytes - dlBytes) / speed);
          }
          var dlShards2 = shards.filter(function(s) { return s.download || s.local; });
          var segmentCount = Math.max(dlShards2.length, shardCount);
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

        // Per-shard download bars (only for small-medium models)
        var perShardDlHtml = '';
        if (isDownloading && shards.length > 0 && shardCount <= 20) {
          var dlShardBars = shards.filter(function(s) {
            return s.download && s.download.state === 'Downloading';
          });
          if (dlShardBars.length > 1) {
            perShardDlHtml = '<div class="per-shard-dl">';
            dlShardBars.forEach(function(s) {
              var pct2 = s.download.progress_pct || 0;
              var bytes = s.download.downloaded_bytes || 0;
              var total = s.download.total_bytes || s.size_bytes || 0;
              perShardDlHtml += '<div class="per-shard-dl-row">' +
                '<span class="per-shard-dl-label">Shard ' + s.index + '</span>' +
                '<div class="per-shard-dl-bar"><div class="per-shard-dl-fill" style="width:' + pct2 + '%"></div></div>' +
                '<span class="per-shard-dl-pct">' + formatBytes(bytes) + '/' + formatBytes(total) + ' (' + pct2 + '%)</span>' +
                '</div>';
            });
            perShardDlHtml += '</div>';
          }
        }

        // Footer meta info
        var footerMeta = [];
        footerMeta.push(formatBytes(m.total_size_bytes || 0));
        if (shardCount > 0) footerMeta.push(shardCount + (shardCount === 1 ? ' shard' : ' shards'));
        if (m.estimated_vram_mb) footerMeta.push('~' + formatMB(m.estimated_vram_mb) + ' VRAM');
        if (m.peers_hosting > 0) footerMeta.push(m.peers_hosting + ' peer' + (m.peers_hosting !== 1 ? 's' : ''));
        else if (hostedShards > 0) footerMeta.push('<span style="color:var(--orange)">Local only</span>');

        // Missing files warning
        var fileIndicators = '';
        if (hostedShards > 0 || isDownloading) {
          var hasManifest = m.has_manifest !== false;
          var hasHeader = m.has_header !== false;
          if (!hasManifest || !hasHeader) {
            var missingFiles = [];
            if (!hasManifest) missingFiles.push('manifest');
            if (!hasHeader) missingFiles.push('header');
            fileIndicators = '<span style="color:var(--orange);font-size:0.7rem" title="Missing: ' + missingFiles.join(', ') + '">&#9888; Missing ' + missingFiles.join(' + ') + '</span>';
          }
        }

        // Action buttons (go in footer)
        var actionHtml = '';
        if (m.status === 'loaded') {
          // Active — no Use button needed, show Chat hint
          actionHtml = '<button class="btn btn-sm btn-outline" data-unload-model="' + escapeHtml(m.id) + '">Unload</button>';
        } else if (isReady) {
          actionHtml = '<button class="btn btn-sm btn-primary" data-select-model="' + escapeHtml(m.id) + '">Use</button>';
        } else if (isDownloading) {
          actionHtml = '<button class="shard-cancel-btn" data-cancel-download="' + escapeHtml(m.id) + '" title="Cancel download">&times; Cancel</button>';
        } else if (m.source === 'network' || m.status === 'available' || m.status === 'partial') {
          actionHtml = '<button class="btn btn-sm" data-request-model="' + escapeHtml(m.id) + '">Download</button>';
        }

        var removeHtml = '';
        if (hostedShards > 0 && !isDownloading) {
          removeHtml = '<button class="model-remove-btn" data-remove-model="' + escapeHtml(m.id) + '">Remove</button>';
        }

        var name = formatModelDisplayName(m.name || m.id);

        // Creator/family icon for swarm model card
        var creatorIconHtml = providerIconHtml(modelIconKey(m.id), 14);

        // ── Card HTML ───────────────────────────────────────────────────────
        card.innerHTML =
          // Title bar
          '<div class="model-card-title">' +
            '<div class="model-card-name-row">' +
              creatorIconHtml +
              '<span class="model-name" title="' + escapeHtml(m.id) + '">' + escapeHtml(name) + '</span>' +
              encBadge + sourceLabel + trustBadge +
            '</div>' +
            '<div class="model-card-controls">' +
              statusHtml + metaBtnHtml + gearHtml +
            '</div>' +
          '</div>' +
          // Shard body
          '<div class="model-card-shards">' +
            shardHtml + progressHtml + perShardDlHtml +
          '</div>' +
          // Footer: stats + actions
          '<div class="model-card-footer">' +
            '<div class="model-card-meta">' +
              footerMeta.map(function(p) { return '<span>' + p + '</span>'; }).join('') +
              (fileIndicators ? '<span>' + fileIndicators + '</span>' : '') +
            '</div>' +
            '<div class="model-card-actions">' + actionHtml + removeHtml + '</div>' +
          '</div>' +
          '<div class="gguf-metadata-panel hidden" data-meta-panel="' + escapeHtml(m.id) + '"></div>';

        if (swarmBody) swarmBody.appendChild(card);
      });

      // Cloud provider models — one compact card per provider
      if (hasCloud) {
        var providerLabels = {
          openai: 'OpenAI', anthropic: 'Anthropic', deepseek: 'DeepSeek',
          mistral: 'Mistral', groq: 'Groq', nvidia_nim: 'NVIDIA NIM',
          cerebras: 'Cerebras', sambanova: 'SambaNova', fireworks: 'Fireworks AI',
          together: 'Together AI', deepinfra: 'DeepInfra', moonshot: 'Moonshot (Kimi)'
        };
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

        // Non-chat model patterns — these get deprioritized in default sort
        var _nonChatPattern = /dall-e|tts|whisper|embed|moderation|davinci-\d|babbage-\d|text-embedding|audio/i;

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
              var la = sa ? sa.latency_ms : 99999, lb = sb ? sb.latency_ms : 99999;
              return la - lb;
            });
          } else if (sortBy === 'popular') {
            // Newest first (by created timestamp), non-chat models pushed to end
            sorted.sort(function(a, b) {
              var aNon = _nonChatPattern.test(a.id) ? 1 : 0;
              var bNon = _nonChatPattern.test(b.id) ? 1 : 0;
              if (aNon !== bNon) return aNon - bNon;
              var ca = (a.meta && a.meta.created) || 0;
              var cb = (b.meta && b.meta.created) || 0;
              if (ca !== cb) return cb - ca; // newest first
              var na = (a.name || a.id).toLowerCase(), nb = (b.name || b.id).toLowerCase();
              return na < nb ? -1 : na > nb ? 1 : 0;
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

        // Helper: render a single model row
        function renderCloudRow(cm) {
          var ctx = getCtxLen(cm);
          var ctxStr = ctx > 0 ? (ctx >= 1000 ? Math.round(ctx / 1000) + 'K' : ctx.toString()) : '';
          var pingHtml = modelStatusBadgeHtml(cm.id);
          return '<div class="cloud-model-row" data-select-cloud="' + escapeHtml(cm.id) + '" title="' + escapeHtml(cm.id) + '">' +
            '<span class="cloud-model-row-name">' + escapeHtml(cm.name || cm.id) + '</span>' +
            (ctxStr ? '<span class="cloud-model-row-ctx">' + ctxStr + '</span>' : '<span class="cloud-model-row-ctx"></span>') +
            '<span class="cloud-model-row-ping">' + pingHtml + '</span>' +
            '</div>';
        }

        // Helper: render all rows into the list container
        function renderRowsInto(container, models) {
          container.innerHTML = models.length > 0
            ? models.map(renderCloudRow).join('')
            : '<div class="cloud-model-empty">No models match</div>';
        }

        // Cloud section wrapper (collapsible)
        var providerCount = Object.keys(byProvider).length;
        var cloudSection = document.createElement('details');
        cloudSection.className = 'models-section';
        cloudSection.open = true;
        var cloudMeta = providerCount + ' provider' + (providerCount !== 1 ? 's' : '') +
          ' \u00b7 ' + cloudModels.length + ' models';
        cloudSection.innerHTML = '<summary class="models-section-header">' +
          '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" aria-hidden="true" class="models-section-logo" style="flex-shrink:0"><path d="M18 10h-1.26A8 8 0 1 0 9 20h9a5 5 0 0 0 0-10z" fill="var(--accent)"/></svg>' +
          '<span class="models-section-title">Cloud Providers</span>' +
          '<span class="models-section-count">' + cloudMeta + '</span>' +
          '</summary>';
        var cloudBody = document.createElement('div');
        cloudBody.className = 'models-section-body';
        cloudSection.appendChild(cloudBody);
        list.appendChild(cloudSection);

        Object.keys(byProvider).forEach(function(p) {
          var pLabel = providerLabels[p] || p;
          var pModels = byProvider[p];
          var sorted = sortCloudModels(pModels, 'popular');
          var filterId = 'cloud-filter-' + p;
          var sortId = 'cloud-sort-' + p;
          var listId = 'cloud-list-wrap-' + p;

          var card = document.createElement('div');
          card.className = 'model-card cloud-model';
          card.setAttribute('data-provider', p);

          var cardIconHtml = providerIconHtml(p, 18);
          card.innerHTML =
            '<div class="cloud-card-header">' +
              '<span class="cloud-provider-name">' + (cardIconHtml ? cardIconHtml + ' ' : '') + escapeHtml(pLabel) + '</span>' +
              '<span>' +
                '<span class="badge badge-cloud">' + pModels.length + ' model' + (pModels.length !== 1 ? 's' : '') + '</span>' +
                '<span class="cloud-status-ok">\u25cf Connected</span>' +
              '</span>' +
            '</div>' +
            '<div class="cloud-card-controls">' +
              '<input type="text" class="cloud-model-filter" id="' + filterId + '" placeholder="Search models\u2026" autocomplete="off">' +
              '<select class="cloud-model-sort" id="' + sortId + '">' +
                '<option value="popular">Newest</option>' +
                '<option value="az">A\u2013Z</option>' +
                '<option value="ctx-desc">Context \u2193</option>' +
                '<option value="ctx-asc">Context \u2191</option>' +
                '<option value="avail">Ping \u2193</option>' +
              '</select>' +
            '</div>' +
            '<div class="cloud-model-list" id="' + listId + '"></div>' +
            '<div class="cloud-card-note">Requests routed to ' + escapeHtml(pLabel) + ' API \u2014 not shared on the swarm network</div>';

          cloudBody.appendChild(card);

          var listContainer = document.getElementById(listId);
          if (listContainer) renderRowsInto(listContainer, sorted);

          // Probe first 20 models for ping
          var visibleIds = sorted.slice(0, 20).map(function(cm) { return cm.id; });
          setTimeout(function() { probeModelStatus(visibleIds); }, 500);

          // Wire filter + sort
          var filterEl = document.getElementById(filterId);
          var sortEl = document.getElementById(sortId);
          var refreshRows = function() {
            var query = filterEl ? filterEl.value.toLowerCase().trim() : '';
            var sortBy = sortEl ? sortEl.value : 'popular';
            var filtered = query ? pModels.filter(function(cm) {
              var text = ((cm.name || '') + ' ' + cm.id + ' ' + (cm.meta && cm.meta.owned_by ? cm.meta.owned_by : '')).toLowerCase();
              return text.indexOf(query) !== -1;
            }) : pModels;
            var s = sortCloudModels(filtered, sortBy);
            if (listContainer) renderRowsInto(listContainer, s);
            // Probe newly visible models
            probeModelStatus(s.slice(0, 20).map(function(cm) { return cm.id; }));
          };
          if (filterEl) {
            filterEl.addEventListener('input', refreshRows);
            filterEl.addEventListener('keyup', refreshRows);
            filterEl.addEventListener('paste', function() { setTimeout(refreshRows, 0); });
          }
          if (sortEl) sortEl.addEventListener('change', function() {
            refreshRows();
            if (sortEl.value === 'avail') probeModelStatus(pModels.map(function(cm) { return cm.id; }).slice(0, 40));
          });
        });
        // Apply cached probe results to newly rendered rows
        if (Object.keys(modelStatus).length > 0) updateModelStatusBadges();
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
                etaStr = formatEta((acq.total_bytes - dlBytes) / speed);
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

          // Update shard summary if present
          var summaryEl = document.querySelector('[data-model-summary="' + safeId + '"]');
          if (summaryEl && shardDetails.length > 0) {
            var summParts = [];
            if (localCount > 0) summParts.push('<span class="shard-sum-item shard-sum-local"><span class="shard-sum-dot"></span>' + localCount + ' local</span>');
            if (peerCount > 0) summParts.push('<span class="shard-sum-item shard-sum-peer"><span class="shard-sum-dot"></span>' + peerCount + ' peer' + (peerCount !== 1 ? 's' : '') + '</span>');
            if (dlCount > 0) summParts.push('<span class="shard-sum-item shard-sum-dl"><span class="shard-sum-dot"></span>' + dlCount + ' downloading</span>');
            if (peerDlCount > 0) summParts.push('<span class="shard-sum-item shard-sum-peer-dl"><span class="shard-sum-dot"></span>' + peerDlCount + ' peer DL</span>');
            if (queuedCount > 0) summParts.push('<span class="shard-sum-item shard-sum-queued"><span class="shard-sum-dot"></span>' + queuedCount + ' queued</span>');
            if (missingCount > 0) summParts.push('<span class="shard-sum-item shard-sum-missing"><span class="shard-sum-dot"></span>' + missingCount + ' missing</span>');
            summaryEl.innerHTML = summParts.join('');
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


    _peersExpanded: false,

    renderPeerItem: function(p) {
      var div = document.createElement('div');
      div.style.cssText = 'padding:6px 10px;background:var(--bg-tertiary);border-radius:var(--radius);border:1px solid var(--border);margin-bottom:4px;display:flex;align-items:center;gap:8px;font-size:0.8rem';
      var statusDot = '<span class="status-dot ' + (p.healthy ? 'online' : 'degraded') + '"></span>';
      var lanTag = p.is_lan_peer ? '<span class="lan-badge">LAN</span>' : '';
      var peerLabel = p.nickname ? escapeHtml(p.nickname) + ' <span class="text-muted mono" style="font-size:0.65rem">(' + escapeHtml(p.node_id || '').substring(0, 8) + ')</span>' : '<span class="mono">' + escapeHtml(p.node_id || 'unknown').substring(0, 16) + '</span>';
      var gpu = p.gpu ? '<span class="text-muted" style="font-size:0.7rem;margin-left:auto">' + escapeHtml(p.gpu) + '</span>' : '';
      div.innerHTML = statusDot + lanTag + peerLabel + gpu;
      return div;
    },

    loadNetworkData: async function() {
      var PEER_LIMIT = 5;
      try {
        var resp = await fetch('/api/admin/peers');
        var peers = await resp.json();
        var list = document.getElementById('peers-list');
        var summary = document.getElementById('peers-summary');
        var overflow = document.getElementById('peers-overflow');
        var pLoading = document.getElementById('peers-loading');
        if (pLoading) pLoading.remove();

        if (peers && peers.length > 0) {
          var lanCount = peers.filter(function(p) { return p.is_lan_peer; }).length;
          var healthyCount = peers.filter(function(p) { return p.healthy; }).length;
          if (summary) {
            summary.textContent = peers.length + ' peer' + (peers.length !== 1 ? 's' : '') +
              (lanCount > 0 ? ' \u00B7 ' + lanCount + ' LAN' : '') +
              ' \u00B7 ' + healthyCount + ' healthy';
          }

          list.innerHTML = '';
          var showAll = dashboard._peersExpanded;
          var visible = showAll ? peers : peers.slice(0, PEER_LIMIT);
          visible.forEach(function(p) {
            list.appendChild(dashboard.renderPeerItem(p));
          });

          if (overflow) {
            if (peers.length > PEER_LIMIT && !showAll) {
              overflow.style.display = '';
              var btn = document.getElementById('btn-show-all-peers');
              if (btn) btn.textContent = 'Show all ' + peers.length + ' peers';
            } else {
              overflow.style.display = 'none';
            }
          }
        } else {
          if (summary) summary.textContent = '';
          list.innerHTML = '<div class="text-muted" style="font-size:0.85rem">' + I18n.t('network.no_peers_yet') + '</div>';
          if (overflow) overflow.style.display = 'none';
        }
      } catch (e) {
        var list = document.getElementById('peers-list');
        var pLoading2 = document.getElementById('peers-loading');
        if (pLoading2) pLoading2.remove();
        if (list) list.innerHTML = '<div class="text-muted" style="font-size:0.85rem">' + I18n.t('network.no_peers_yet') + '</div>';
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
              showToast('Download is slow (' + formatSpeed(speed) + ') \u2014 this can happen with popular models. It will keep going.', 'warning', 10000);
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
      // Don't re-add progress for cancelled downloads
      if (!activeAcquisitions[modelId]) return;
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
        '<span class="text-muted">Downloading model data</span>' +
        '<span style="display:flex;align-items:center;gap:8px">' +
          '<span class="mono dl-progress-text">' + formatBytes(dlBytes) + ' / ' + formatBytes(totalBytes) + ' (' + pct + '%)' + speedStr + '</span>' +
          '<button class="btn btn-sm" style="padding:1px 6px;font-size:0.7rem;line-height:1.2" data-cancel-download="' + escapeHtml(modelId) + '" title="Cancel download">&times; Cancel</button>' +
        '</span>' +
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

          // VRAM fit tags — tiered: shard (any participation), boomerang (request from this node), all shards
          var fitsTag = '';
          var shardSizeStr = repo.est_shard_size ? formatBytes(repo.est_shard_size) : '';
          var boomerangSizeStr = repo.est_boomerang_size ? formatBytes(repo.est_boomerang_size) : '';
          if (repo.fits_boomerang) {
            fitsTag = '<span style="color:var(--green)" title="First+last shard fit VRAM (~' + boomerangSizeStr + ') — can run requests from this node via boomerang routing">&#9989; Run locally</span>';
          } else if (repo.fits_shard) {
            fitsTag = '<span style="color:var(--cyan)" title="Individual shards fit VRAM (~' + shardSizeStr + '/shard) — can participate in swarm inference for this model">&#128279; Can host shards</span>';
          } else if (repo.fits_vram === false && variants.length > 0) {
            fitsTag = '<span style="color:var(--orange)" title="Even individual shards may exceed your available VRAM">&#9888; Exceeds VRAM</span>';
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
            '<button class="btn btn-sm btn-primary" data-hf-download="' + escapeHtml(repo.repo_id) + '" data-hf-variant="' + safeKey + '">Download</button>' +
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
        ui.showBanner('info', 'Checking model availability...');
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
          showToast('Download started — model data will be ready soon', 'success');
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

      // Language picker — sync with I18n
      var langSelect = document.getElementById('settings-language');
      if (langSelect && typeof I18n !== 'undefined') {
        langSelect.value = I18n.getLang() || 'en';
        langSelect.addEventListener('change', function() {
          I18n.setLang(this.value);
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
      } catch (e) {
        ui.showBanner('error', 'Failed to load settings: ' + (e.message || 'network error'));
      }
      // Load API key and provider status
      settings._apiKeyPromise = settings.loadApiKey();
      settings.loadProviders();
    },

    _apiKeyFull: '',
    _apiKeyPromise: null,

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
        var resp = await authFetch('/api/admin/providers');
        var data = await resp.json();
        if (data.providers) {
          var anyConfigured = false;
          data.providers.forEach(function(p) {
            if (p.configured) anyConfigured = true;
            var badge = document.getElementById('provider-status-' + p.name);
            if (badge) {
              if (p.configured && p.source === 'env') {
                badge.textContent = '\u2713 From .env';
                badge.className = 'badge provider-badge-active';
                badge.title = 'Loaded from environment variable or .env file';
              } else if (p.configured) {
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
        // Set key source dropdown
        if (data.key_source) {
          var sel = document.getElementById('provider-key-source');
          if (sel) sel.value = data.key_source;
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
          if (friendlyErr.length > 200) friendlyErr = friendlyErr.substring(0, 200) + '…';
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
        document.getElementById('hw-vram').textContent = setup.hwData.gpu_vram_mb ? formatMB(setup.hwData.gpu_vram_mb) : 'N/A';
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
            '2. Connect with others using Network Code to share AI models<br>' +
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
      if (typeof NeuralBg !== 'undefined') NeuralBg.setHealth(1.0);
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
          var text = 'Pruned shard ' + escapeHtml(String(d.shard_index)) + ' of ' + escapeHtml(d.model_name || d.model_id) +
            ' \u2014 ' + escapeHtml(String(d.holder_count_before)) + '\u2192' + escapeHtml(String(d.holder_count_after)) + ' holders (freed ' + escapeHtml(freed) + ')';
          showPruneToast(text);
          // models_changed event from prune will trigger refresh below
        } else if (msg.type === 'system_notification') {
          var n = msg.data;
          var level = n.level === 'error' ? 'error' : (n.level === 'warn' ? 'warning' : 'info');
          showToast(n.title + ': ' + n.message, level, 10000);
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
      if (typeof NeuralBg !== 'undefined') NeuralBg.setHealth(0.3);
      if (wsWasConnected) {
        showWsBanner('disconnected', 'Lost connection to SwarmLLM \u2014 reconnecting...');
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
      loadModeIndicator();
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
  var providerIcons = (function() {
    var icons = {};
    ['anthropic','openai','deepseek','mistral','groq','nvidia_nim',
     'cerebras','sambanova','fireworks','together','deepinfra','moonshot'].forEach(function(p) {
      var url = providerIconUrl(p);
      icons[p] = url ? '<img src="' + url + '" width="18" height="18" alt="" class="provider-icon" style="display:block">' : '';
    });
    return icons;
  }());

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
      var isError = (h.status === 'auth_error' || h.status === 'down' || h.status === 'error');
      badge.className = 'provider-badge' + (h.status === 'up' ? ' badge-active' : '') + (isError ? ' badge-error' : '');
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
        dotClass = 'dot-error';
        latencyText = 'Key Invalid';
      } else if (h.status === 'overloaded') {
        dotClass = 'dot-ok';
        latencyText = 'Busy';
      } else {
        dotClass = 'dot-error';
        latencyText = 'Down';
      }
      var iconHtml = providerIcons[p] || '';
      var name = providerDisplayNames[p] || p;
      badge.innerHTML = '<span class="pb-icon">' + iconHtml + '</span>' +
        '<span class="pb-name">' + escapeHtml(name) + '</span>' +
        '<span class="pb-dot ' + dotClass + '"></span>' +
        (latencyText ? '<span class="pb-latency">' + escapeHtml(latencyText) + '</span>' : '');
      badge.title = name + ': ' + h.status + (h.detail ? ' — ' + h.detail : '') + (h.latency_ms ? ' (' + h.latency_ms + 'ms)' : '');
      if (isError) {
        badge.style.cursor = 'pointer';
        badge.addEventListener('click', function() { ui.openSettings(true); });
      }
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
      try { sessionStorage.setItem('swarmllm_model_status', JSON.stringify(modelStatus)); } catch (e) {}
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
    // Update cloud model rows on dashboard
    document.querySelectorAll('.cloud-model-row[data-select-cloud]').forEach(function(row) {
      var modelId = row.getAttribute('data-select-cloud');
      var pingEl = row.querySelector('.cloud-model-row-ping');
      if (pingEl) pingEl.innerHTML = modelStatusBadgeHtml(modelId);
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
        var pmResp = await authFetch('/api/admin/provider-models');
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
        var localItems = [];
        var swarmItems = [];
        readyModels.forEach(function(m) {
          var displayName = formatModelDisplayName(m.name || m.id);
          var isDistributed = m.shard_count > 0 && (m.hosted_shards || 0) < m.shard_count;
          var item = { id: m.id, name: displayName.length > 40 ? displayName.substring(0, 40) + '...' : displayName, group: isDistributed ? 'swarm' : 'local', encrypted: !!m.encrypted_pipeline };
          if (isDistributed) { swarmItems.push(item); } else { localItems.push(item); }
        });
        if (localItems.length > 0) {
          groups.push({ key: 'local', label: 'On this computer', items: localItems });
          _modelDropdownData = _modelDropdownData.concat(localItems);
        }
        if (swarmItems.length > 0) {
          groups.push({ key: 'swarm', label: 'Swarm network', items: swarmItems });
          _modelDropdownData = _modelDropdownData.concat(swarmItems);
        }
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
      // Re-render chat header and session list now that model data is available
      // (fixes wrong source badges on sessions loaded before model fetch completes)
      if (typeof chat !== 'undefined' && chat.updateChatHeader) chat.updateChatHeader();
      if (typeof chat !== 'undefined' && chat.renderSessionList) chat.renderSessionList();
    } catch (e) {
      ui.showBanner('error', 'Failed to load models: ' + (e.message || 'network error'));
    }
  }

  function renderModelDropdown(groups, hasAny) {
    var list = document.getElementById('model-dropdown-list');
    if (!list) return;
    list.innerHTML = '';

    if (!hasAny) {
      list.innerHTML = '<div class="model-dropdown-empty">No models available<br><span style="font-size:0.72rem;color:var(--text-muted)">Click + to download a model, or add a cloud provider in Settings</span></div>';
      return;
    }

    groups.forEach(function(g) {
      var groupEl = document.createElement('div');
      groupEl.className = 'model-dropdown-group';
      groupEl.setAttribute('data-group', g.key);

      var header = document.createElement('div');
      header.className = 'model-dropdown-group-header';
      var groupIconHtml = providerIconHtml(g.key, 14);
      header.innerHTML = '<span class="group-arrow">&#9662;</span>' + (groupIconHtml ? ' ' + groupIconHtml : '') + ' ' + escapeHtml(g.label) + ' <span style="opacity:0.5;font-weight:400">(' + g.items.length + ')</span>';
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
        // Show full model ID on hover so users can distinguish quantizations
        if (item.id !== item.name) el.setAttribute('title', item.id);
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
        // Empty session — just update its model and refresh empty state
        s.model = modelId;
        chat.saveSessions();
        chat.renderMessages();
        chat.updateChatHeader();
        chat.renderSessionList();
      }
    }
  }

  function updateModelDropdownLabel(text) {
    var label = document.getElementById('model-dropdown-label');
    if (!label) return;
    // Check if current model has encrypted pipeline via the dropdown data
    var item = _modelDropdownData.find(function(m) { return m.name === text || m.id === text; });
    var enc = item && item.encrypted;
    label.innerHTML = escapeHtml(text) + (enc ? ' <span class="badge-encrypted" title="Encrypted pipeline active">&#128274;</span>' : '');
    // Show full model ID on hover so users can see quantization details
    var trigger = document.getElementById('model-dropdown-trigger');
    if (trigger && item) trigger.title = item.id;
    // Highlight dropdown when no model is selected
    if (trigger) trigger.classList.toggle('no-model', !currentModel);
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
        ui.showBanner('warning', data.message || 'Model download unavailable');
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
          // Reset any shard cells stuck in downloading state
          card.querySelectorAll('.shard-cell.downloading, .shard-cell.verifying').forEach(function(cell) {
            var idx = cell.getAttribute('data-shard-index') || cell.textContent;
            cell.className = 'shard-cell missing';
            cell.textContent = idx;
            cell.style.removeProperty('--dl-pct');
          });
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
    banner.style.cssText = 'position:fixed;top:0;left:0;right:0;z-index:10000;background:var(--yellow, #eab308);color:var(--bg-primary, #0a0e14);padding:0.6rem 1rem;display:flex;align-items:center;justify-content:center;gap:1rem;font-size:0.85rem;font-weight:500;box-shadow:0 2px 8px rgba(0,0,0,0.3)';
    var text = 'Update available: v' + escapeHtml(data.current_version) + ' \u2192 v' + escapeHtml(data.latest_version);
    banner.innerHTML = '<span>' + text + '</span>';
    if (data.downloaded) {
      var applyBtn = document.createElement('button');
      applyBtn.textContent = 'Apply & Restart';
      applyBtn.style.cssText = 'background:var(--bg-primary, #0a0e14);color:var(--yellow, #eab308);border:none;border-radius:4px;padding:0.3rem 0.8rem;cursor:pointer;font-size:0.8rem;font-weight:600';
      applyBtn.onclick = async function() {
        applyBtn.disabled = true;
        applyBtn.textContent = 'Applying...';
        try {
          var resp = await authFetch('/api/admin/update/apply', { method: 'POST' });
          if (resp.ok) {
            banner.querySelector('span').textContent = 'Update applied! Restart SwarmLLM to use v' + escapeHtml(data.latest_version);
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
      dlBtn.style.cssText = 'background:var(--bg-primary, #0a0e14);color:var(--yellow, #eab308);border:none;border-radius:4px;padding:0.3rem 0.8rem;cursor:pointer;font-size:0.8rem;font-weight:600';
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
                banner.querySelector('span').textContent = 'Update applied! Restart SwarmLLM to use v' + escapeHtml(data.latest_version);
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
            ui.showBanner('success', 'Downloading model part ' + (idx + 1));
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
        else if (stateName === 'awaiting_manifest') { stateLabel = 'Preparing download...'; stateClass = 'waiting'; }
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
          etaStr = formatEta(dl.eta_secs);
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
            right += ' \u00b7 ETA ' + formatEta((totalBytes - dlBytes) / speed);
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

    // Fetch current policy + encrypted pipeline status
    var policy = { enabled: true, max_shards: 0, prune_enabled: true };
    var encStatus = { encrypted_pipeline: false, ready: false, has_first_shard: false, has_last_shard: false, shard_count: 0 };
    try {
      var [amResp, encResp] = await Promise.all([
        authFetch('/api/admin/models/' + encodeURIComponent(modelId) + '/auto-manage'),
        authFetch('/api/admin/models/' + encodeURIComponent(modelId) + '/encrypted-pipeline'),
      ]);
      if (amResp.ok) policy = await amResp.json();
      if (encResp.ok) encStatus = await encResp.json();
    } catch (e) {
      ui.showBanner('error', 'Could not load model policy');
    }

    var encReadyClass = encStatus.ready ? 'text-success' : 'text-warning';
    var encReadyText = encStatus.ready ? 'Ready (has first + last shard)' :
      'Missing: ' + (!encStatus.has_first_shard ? 'first shard ' : '') + (!encStatus.has_last_shard ? 'last shard' : '');
    var encDisabled = !encStatus.ready ? ' disabled' : '';
    var encOverheadNote = encStatus.shard_count <= 2
      ? '<span class="text-warning" style="font-size:0.65rem">&#9888; ' + encStatus.shard_count + '-shard model = fully local (no distributed offloading)</span>'
      : '<span class="text-muted" style="font-size:0.65rem">Adds ~1 extra RTT/token. No remote node sees plaintext.</span>';

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
      '<hr style="margin:0.3rem 0;border-color:var(--border)">' +
      '<div class="am-row" style="flex-direction:column;gap:0.2rem">' +
        '<label><input type="checkbox" id="am-encrypted-' + escapeHtml(modelId) + '"' +
          (encStatus.encrypted_pipeline ? ' checked' : '') + encDisabled +
          '> &#128274; Encrypted pipeline</label>' +
        '<span class="' + encReadyClass + '" style="font-size:0.65rem">' + encReadyText + '</span>' +
        encOverheadNote +
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
    var encryptedEl = document.getElementById('am-encrypted-' + safeId);
    if (!enabledEl || !maxEl) return;

    try {
      // Save auto-manage policy
      var amResp = await authFetch('/api/admin/models/' + encodeURIComponent(modelId) + '/auto-manage', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          enabled: enabledEl.checked,
          max_shards: parseInt(maxEl.value, 10) || 0,
          prune_enabled: pruneEl ? pruneEl.checked : true,
        }),
      });

      // Save encrypted pipeline toggle (if checkbox exists and not disabled)
      var encErr = null;
      if (encryptedEl && !encryptedEl.disabled) {
        var encResp = await authFetch('/api/admin/models/' + encodeURIComponent(modelId) + '/encrypted-pipeline', {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ enabled: encryptedEl.checked }),
        });
        if (!encResp.ok) {
          var encData = await encResp.json().catch(function() { return {}; });
          encErr = encData.error ? encData.error.message : 'Encrypted pipeline save failed';
        }
      }

      if (amResp.ok && !encErr) {
        ui.showBanner('success', 'Model policy saved');
        var card = document.querySelector('[data-model-id="' + modelId + '"]');
        var panel = card ? card.querySelector('.auto-manage-panel') : null;
        if (panel) panel.remove();
      } else {
        var errMsg = encErr || '';
        if (!amResp.ok) {
          var errData = await amResp.json().catch(function() { return {}; });
          errMsg = errData.error ? errData.error.message : 'Save failed';
        }
        ui.showBanner('error', errMsg);
      }
    } catch (e) {
      ui.showBanner('error', 'Save failed: ' + e.message);
    }
  }

  async function removeModel(modelId) {
    if (!confirm('Remove all local shards for ' + modelId + '? This cannot be undone.')) return;
    try {
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

  function formatEta(seconds) {
    var s = Math.round(seconds);
    if (s >= 3600) return Math.floor(s / 3600) + 'h ' + Math.floor((s % 3600) / 60) + 'm';
    if (s >= 60) return Math.floor(s / 60) + 'm ' + (s % 60) + 's';
    return s + 's';
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

  function getModelSource(modelId) {
    if (!modelId) return 'local';
    var match = _modelDropdownData.find(function(m) { return m.id === modelId; });
    if (!match) return 'local';
    if (match.group === 'local') return 'local';
    if (match.group === 'swarm') return 'swarm';
    return 'cloud';
  }

  function applyMessageGrouping(container) {
    var msgs = Array.prototype.slice.call(container.querySelectorAll('.chat-msg'));
    if (!msgs.length) return;
    var i = 0;
    while (i < msgs.length) {
      var role = msgs[i].classList.contains('user') ? 'user' : 'assistant';
      var j = i;
      while (j + 1 < msgs.length) {
        var nextRole = msgs[j + 1].classList.contains('user') ? 'user' : 'assistant';
        if (nextRole !== role) break;
        j++;
      }
      var size = j - i + 1;
      for (var k = i; k <= j; k++) {
        msgs[k].classList.remove('group-solo', 'group-first', 'group-mid', 'group-last');
        if (size === 1) {
          msgs[k].classList.add('group-solo');
        } else if (k === i) {
          msgs[k].classList.add('group-first');
        } else if (k === j) {
          msgs[k].classList.add('group-last');
        } else {
          msgs[k].classList.add('group-mid');
        }
      }
      i = j + 1;
    }
  }

  function appendMessageToDOM(role, content, isHtml, opts) {
    opts = opts || {};
    var container = document.getElementById('chat-messages');
    var empty = document.getElementById('chat-empty');
    if (empty) empty.style.display = 'none';

    var div = document.createElement('div');
    div.className = 'chat-msg ' + role;

    // Add source indicator for assistant messages
    var sourceHtml = '';
    if (role === 'assistant') {
      var session = currentSessionId && sessions[currentSessionId] ? sessions[currentSessionId] : null;
      var modelId = opts.model || (session ? session.model : '') || currentModel || '';
      var source = getModelSource(modelId);
      div.classList.add('source-' + source);
      var sourceLabel = source === 'local' ? 'Your PC' : source === 'cloud' ? 'Cloud' : 'Network';
      sourceHtml = '<span class="msg-source-badge source-' + source + '">' + sourceLabel + '</span>';
    }

    // Avatar slot — visible only in messenger mode
    var avatarEl = document.createElement('div');
    avatarEl.className = 'msg-avatar';
    avatarEl.setAttribute('aria-hidden', 'true');
    if (role === 'assistant') {
      var _sess = currentSessionId && sessions[currentSessionId] ? sessions[currentSessionId] : null;
      var _mid = opts && opts.model ? opts.model : (_sess ? _sess.model : '') || currentModel || '';
      var _avatarProvider = (_mid && _modelDropdownData.find(function(m) { return m.id === _mid; }) || {}).group || null;
      var _iconKey = (_avatarProvider && _ICON_MAP[_avatarProvider]) ? _avatarProvider : modelIconKey(_mid);
      var _iconUrl = _iconKey ? providerIconUrl(_iconKey) : null;
      if (_iconUrl) {
        avatarEl.innerHTML = '<img src="' + _iconUrl + '" width="16" height="16" alt="" class="provider-icon provider-avatar-icon" style="display:block">';
      } else {
        avatarEl.textContent = 'AI';
      }
    } else {
      avatarEl.innerHTML = '<img src="/static/icons/swarm.svg" alt="" style="width:16px;height:16px;display:block;">';
    }
    div.appendChild(avatarEl);

    // Bubble wrapper — used by messenger mode; transparent pass-through in linear mode
    var bubble = document.createElement('div');
    bubble.className = 'msg-bubble';

    var label = role === 'user' ? 'You' : 'Assistant';
    bubble.innerHTML = '<div class="msg-role">' + label + sourceHtml + '</div><div class="msg-content"></div>';
    if (isHtml) {
      bubble.querySelector('.msg-content').innerHTML = content;
    } else {
      bubble.querySelector('.msg-content').textContent = content;
    }

    // Add action buttons for assistant messages
    if (role === 'assistant') {
      var actions = document.createElement('div');
      actions.className = 'msg-actions';
      actions.innerHTML = '<button class="msg-action-btn" data-action="copy" title="Copy this response">Copy</button>' +
        '<button class="msg-action-btn" data-action="compare" title="Ask other models the same question">Try other models</button>';
      bubble.appendChild(actions);
    }

    div.appendChild(bubble);
    container.appendChild(div);
    applyMessageGrouping(container);
    chat.scrollToBottom();
    return div;
  }

  function createEmptyState() {
    var div = document.createElement('div');
    div.className = 'chat-empty';
    div.id = 'chat-empty';

    // Resolve current model name and status
    var modelName = '';
    var modelData = null;
    if (currentModel) {
      var item = _modelDropdownData.find(function(m) { return m.id === currentModel; });
      modelName = item ? item.name : currentModel;
      modelData = (window._lastModelsData || []).find(function(m) { return m.id === currentModel; });
    }

    var title = modelName ? 'Chat with ' + escapeHtml(modelName) : 'Chat with AI';
    var icon = '&#11088;';

    // Encryption / routing info
    var encHint = '';
    if (modelData && modelData.encrypted_pipeline && modelData.shard_count > 1) {
      var isFullLocal = modelData.hosted_shards === modelData.shard_count;
      if (isFullLocal) {
        icon = '&#128274;';
        encHint = '<div class="chat-empty-hint" style="margin:6px 0;font-size:0.8rem;color:var(--green)">' +
          '&#128274; Encrypted pipeline active \u2014 all shards local, full privacy' +
          '</div>';
      } else {
        icon = '&#128274;';
        encHint = '<div class="chat-empty-hint" style="margin:6px 0;font-size:0.8rem;color:var(--orange)">' +
          '&#128274; Boomerang routing \u2014 first + last shard local, middle shards on peers' +
          '<br><span style="font-size:0.75rem;color:var(--text-muted)">Prompts are encrypted end-to-end. Expect ~2\u20135s extra latency for distributed pipeline setup.</span>' +
          '</div>';
      }
    } else if (modelData && modelData.shard_count > 1 && modelData.hosted_shards < modelData.shard_count) {
      encHint = '<div class="chat-empty-hint" style="margin:6px 0;font-size:0.8rem;color:var(--text-muted)">' +
        '&#127760; Distributed inference \u2014 shards split across peers' +
        '<br><span style="font-size:0.75rem">Enable encrypted pipeline in the model card for end-to-end privacy.</span>' +
        '</div>';
    }

    div.innerHTML = '<div class="chat-empty-icon">' + icon + '</div>' +
      '<div class="chat-empty-title">' + title + '</div>' +
      encHint +
      '<div class="chat-empty-hint" style="margin:8px 0">Type a message below and press <kbd>Enter</kbd> to send</div>' +
      '<div class="chat-empty-hint" style="font-size:0.8rem;margin-top:4px">' +
        (modelName ? '' : 'Pick a model from the dropdown above \u2022 ') +
        '<kbd>Shift+Enter</kbd> for new line</div>';
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
    var words = text.trim().split(/\s+/).length;
    el.textContent = words + ' words';
    if (tokens > 7000) { el.className = 'token-counter danger'; el.title = 'Very long message — some models may not handle this length'; }
    else if (tokens > 3000) { el.className = 'token-counter warn'; el.title = 'Long message — response quality may vary'; }
    else { el.className = 'token-counter'; el.title = 'Message length'; }
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
          tbody.innerHTML = '<tr><td colspan="4" class="text-muted" style="text-align:center;padding:24px">No activity yet. Credits are earned by helping others run AI models.</td></tr>';
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

    // ISO 3166-1 numeric → alpha-2 (comprehensive — covers all 177 Natural Earth countries)
    numToAlpha2: {
      '004':'AF','008':'AL','012':'DZ','024':'AO','032':'AR','036':'AU','040':'AT',
      '044':'BS','050':'BD','056':'BE','064':'BT','068':'BO','070':'BA','072':'BW',
      '076':'BR','084':'BZ','090':'SB','096':'BN','100':'BG','104':'MM','108':'BI',
      '112':'BY','116':'KH','120':'CM','124':'CA','140':'CF','144':'LK','148':'TD',
      '152':'CL','156':'CN','158':'TW','170':'CO','178':'CG','180':'CD','188':'CR',
      '191':'HR','192':'CU','196':'CY','203':'CZ','204':'BJ','208':'DK','214':'DO',
      '218':'EC','222':'SV','226':'GQ','231':'ET','232':'ER','233':'EE','238':'FK',
      '242':'FJ','246':'FI','250':'FR','260':'TF','262':'DJ','266':'GA','268':'GE',
      '270':'GM','275':'PS','276':'DE','288':'GH','296':'KI','300':'GR','304':'GL',
      '320':'GT','324':'GN','328':'GY','332':'HT','340':'HN','344':'HK','348':'HU',
      '352':'IS','356':'IN','360':'ID','364':'IR','368':'IQ','372':'IE','376':'IL',
      '380':'IT','384':'CI','388':'JM','392':'JP','398':'KZ','400':'JO','404':'KE',
      '408':'KP','410':'KR','414':'KW','417':'KG','418':'LA','422':'LB','426':'LS',
      '428':'LV','430':'LR','434':'LY','440':'LT','442':'LU','450':'MG','454':'MW',
      '458':'MY','462':'MV','466':'ML','478':'MR','484':'MX','496':'MN','498':'MD',
      '504':'MA','508':'MZ','512':'OM','516':'NA','524':'NP','528':'NL','540':'NC',
      '548':'VU','554':'NZ','558':'NI','562':'NE','566':'NG','578':'NO','586':'PK',
      '591':'PA','598':'PG','600':'PY','604':'PE','608':'PH','616':'PL','620':'PT',
      '624':'GW','626':'TL','634':'QA','642':'RO','643':'RU','646':'RW','682':'SA',
      '686':'SN','694':'SL','702':'SG','703':'SK','704':'VN','706':'SO','710':'ZA',
      '716':'ZW','724':'ES','728':'SS','729':'SD','732':'EH','740':'SR','752':'SE',
      '756':'CH','760':'SY','762':'TJ','764':'TH','768':'TG','780':'TT','784':'AE',
      '788':'TN','792':'TR','795':'TM','800':'UG','804':'UA','818':'EG','826':'GB',
      '834':'TZ','840':'US','854':'BF','858':'UY','860':'UZ','862':'VE','887':'YE',
      '894':'ZM',
    },

    // Equirectangular projection: lon/lat → SVG coords in viewBox 0 0 1000 500
    projectCoord: function(lon, lat) {
      var x = (lon + 180) / 360 * 1000;
      var y = (90 - lat) / 180 * 500;
      return [Math.round(x * 10) / 10, Math.round(y * 10) / 10];
    },

    // Convert a GeoJSON ring (array of [lon,lat]) to SVG path d string
    // Handles antimeridian crossings (e.g. Russia, Fiji) by breaking with M commands
    ringToPath: function(ring) {
      var parts = [];
      for (var i = 0; i < ring.length; i++) {
        var p = networkMap.projectCoord(ring[i][0], ring[i][1]);
        if (i === 0) {
          parts.push('M' + p[0] + ',' + p[1]);
        } else {
          // Detect antimeridian crossing: longitude jump > 180°
          var lonDiff = Math.abs(ring[i][0] - ring[i - 1][0]);
          if (lonDiff > 180) {
            parts.push('M' + p[0] + ',' + p[1]);
          } else {
            parts.push('L' + p[0] + ',' + p[1]);
          }
        }
      }
      parts.push('Z');
      return parts.join('');
    },

    // Convert a GeoJSON geometry to SVG path d string
    geomToPath: function(geom) {
      var d = '';
      if (geom.type === 'Polygon') {
        for (var i = 0; i < geom.coordinates.length; i++) {
          d += networkMap.ringToPath(geom.coordinates[i]);
        }
      } else if (geom.type === 'MultiPolygon') {
        for (var i = 0; i < geom.coordinates.length; i++) {
          for (var j = 0; j < geom.coordinates[i].length; j++) {
            d += networkMap.ringToPath(geom.coordinates[i][j]);
          }
        }
      }
      return d;
    },

    // Paths populated from TopoJSON at runtime
    paths: {},

    buildSvg: async function() {
      var container = document.getElementById('world-map');
      if (!container) return;

      // Load TopoJSON data
      try {
        var resp = await fetch('/static/data/countries-110m.json');
        var topo = await resp.json();
        var geojson = topojson.feature(topo, topo.objects.countries);
        var features = geojson.features;

        // Build paths from real geographic data
        networkMap.paths = {};
        for (var i = 0; i < features.length; i++) {
          var f = features[i];
          var numId = String(f.id);
          var alpha2 = networkMap.numToAlpha2[numId];
          if (!alpha2) continue; // Skip unmapped territories
          var d = networkMap.geomToPath(f.geometry);
          networkMap.paths[alpha2] = d;
        }
      } catch (e) {
        // TopoJSON load failed — map will be empty
        console.warn('[SwarmLLM] Failed to load map data:', e.message);
      }

      var svg = '<svg viewBox="0 0 1000 500" xmlns="http://www.w3.org/2000/svg" class="world-svg">';
      // Defs — glow filter for active regions
      svg += '<defs>';
      svg += '<filter id="glow-sm"><feGaussianBlur stdDeviation="1.5" result="blur"/><feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge></filter>';
      svg += '<filter id="glow-md"><feGaussianBlur stdDeviation="3" result="blur"/><feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge></filter>';
      svg += '</defs>';
      // Background
      svg += '<rect width="1000" height="500" fill="transparent" rx="4"/>';
      // Grid lines — subtle crosshatch
      for (var x = 0; x <= 1000; x += 50) {
        var op = (x % 100 === 0) ? '0.25' : '0.1';
        svg += '<line x1="' + x + '" y1="0" x2="' + x + '" y2="500" stroke="var(--accent)" stroke-width="0.3" opacity="' + op + '"/>';
      }
      for (var y = 0; y <= 500; y += 50) {
        var opy = (y % 100 === 0) ? '0.25' : '0.1';
        svg += '<line x1="0" y1="' + y + '" x2="1000" y2="' + y + '" stroke="var(--accent)" stroke-width="0.3" opacity="' + opy + '"/>';
      }
      // Equator — faint reference line
      svg += '<line x1="0" y1="250" x2="1000" y2="250" stroke="var(--accent)" stroke-width="0.5" opacity="0.15" stroke-dasharray="8,4"/>';
      // Country paths — outline-focused neon style
      var codes = Object.keys(networkMap.paths);
      for (var i = 0; i < codes.length; i++) {
        var code = codes[i];
        var d = networkMap.paths[code];
        if (!d) continue;
        svg += '<path id="region-' + code + '" d="' + d + '" fill="rgba(59,130,246,0.04)" stroke="rgba(59,130,246,0.3)" stroke-width="0.5" class="map-region" data-code="' + code + '"/>';
      }
      svg += '</svg>';
      container.innerHTML = svg;

      // Add hover tooltip handlers — track mouse position for cursor-following tooltip
      container.querySelectorAll('.map-region').forEach(function(el) {
        el.addEventListener('mouseenter', function(e) { networkMap.showTooltip(e, el.dataset.code); });
        el.addEventListener('mousemove', function(e) { networkMap.moveTooltip(e); });
        el.addEventListener('mouseleave', function() { networkMap.hideTooltip(); });
      });

      networkMap.mapRendered = true;
    },

    refresh: async function() {
      if (!networkMap.mapRendered) await networkMap.buildSvg();
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
          el.style.fill = 'rgba(59,130,246,0.04)';
          el.style.stroke = 'rgba(59,130,246,0.3)';
          el.style.strokeWidth = '0.5';
          el.removeAttribute('filter');
        } else {
          var intensity = Math.max(0.25, n / Math.max(maxCount, 1));
          var fillAlpha = (0.06 + intensity * 0.14).toFixed(2);
          var strokeAlpha = (0.5 + intensity * 0.5).toFixed(2);
          el.style.fill = 'rgba(59,130,246,' + fillAlpha + ')';
          el.style.stroke = 'rgba(100,180,255,' + strokeAlpha + ')';
          el.style.strokeWidth = (0.8 + intensity * 1.2).toFixed(1);
          el.setAttribute('filter', 'url(#glow-md)');
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
          el.style.fill = 'rgba(59,130,246,0.04)';
          el.style.stroke = 'rgba(59,130,246,0.3)';
          el.style.strokeWidth = '0.5';
          el.removeAttribute('filter');
        } else {
          var intensity = Math.max(0.25, n / Math.max(maxCount, 1));
          var fillAlpha = (0.06 + intensity * 0.14).toFixed(2);
          var strokeAlpha = (0.5 + intensity * 0.5).toFixed(2);
          el.style.fill = 'rgba(59,130,246,' + fillAlpha + ')';
          el.style.stroke = 'rgba(100,180,255,' + strokeAlpha + ')';
          el.style.strokeWidth = (0.8 + intensity * 1.2).toFixed(1);
          el.setAttribute('filter', 'url(#glow-md)');
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
      var mapContainer = document.getElementById('world-map-container');
      mapContainer.appendChild(tip);
      networkMap.moveTooltip(event);
      // Fade in
      requestAnimationFrame(function() { tip.classList.add('visible'); });
    },

    moveTooltip: function(event) {
      var tip = document.getElementById('map-tooltip');
      if (!tip) return;
      var mapContainer = document.getElementById('world-map-container');
      var containerRect = mapContainer.getBoundingClientRect();
      var x = event.clientX - containerRect.left + 14;
      var y = event.clientY - containerRect.top - tip.offsetHeight - 10;
      // Keep tooltip within container bounds
      if (x + tip.offsetWidth > containerRect.width - 8) x = event.clientX - containerRect.left - tip.offsetWidth - 14;
      if (y < 4) y = event.clientY - containerRect.top + 18;
      tip.style.left = x + 'px';
      tip.style.top = y + 'px';
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
    // Basic/Advanced toggle removed — always advanced

    // Theme toggle (light / dark / system)
    on('btn-theme-toggle', 'click', function() {
      var THEME_KEY = 'swarmllm_theme';
      var themes = ['dark', 'light', 'system'];
      var icons = { dark: '\u263E', light: '\u2600', system: '\u25D1' };
      var cur = localStorage.getItem(THEME_KEY) || 'dark';
      var next = themes[(themes.indexOf(cur) + 1) % themes.length];
      localStorage.setItem(THEME_KEY, next);
      applyTheme(next);
      var btn = document.getElementById('btn-theme-toggle');
      if (btn) btn.textContent = icons[next] || '\u263E';
    });

    // Language picker dropdown
    (function() {
      var LANGS = [
        ['en','English'],['es','Español'],['fr','Français'],['de','Deutsch'],
        ['pt','Português'],['it','Italiano'],['nl','Nederlands'],['ru','Русский'],
        ['zh','中文'],['ja','日本語'],['ko','한국어'],['ar','العربية'],
        ['tr','Türkçe'],['pl','Polski'],['sv','Svenska'],['th','ไทย'],
        ['hi','हिन्दी'],['vi','Tiếng Việt'],['id','Bahasa Indonesia'],
        ['uk','Українська'],['cs','Čeština']
      ];
      var dropdown = document.getElementById('lang-dropdown');
      var btn = document.getElementById('btn-lang-picker');
      if (!dropdown || !btn) return;
      LANGS.forEach(function(pair) {
        var b = document.createElement('button');
        b.type = 'button';
        b.textContent = pair[1];
        b.dataset.lang = pair[0];
        b.addEventListener('click', function() {
          if (typeof I18n !== 'undefined') I18n.setLang(pair[0]);
          var settingsLang = document.getElementById('settings-language');
          if (settingsLang) settingsLang.value = pair[0];
          var setupLang = document.getElementById('setup-language');
          if (setupLang) setupLang.value = pair[0];
          dropdown.style.display = 'none';
          updateLangDropdownActive();
        });
        dropdown.appendChild(b);
      });
      btn.addEventListener('click', function(e) {
        e.stopPropagation();
        var open = dropdown.style.display !== 'none';
        dropdown.style.display = open ? 'none' : '';
        if (!open) updateLangDropdownActive();
      });
      document.addEventListener('click', function() { dropdown.style.display = 'none'; });
      dropdown.addEventListener('click', function(e) { e.stopPropagation(); });

      function updateLangDropdownActive() {
        var cur = (typeof I18n !== 'undefined') ? I18n.getLang() : 'en';
        dropdown.querySelectorAll('button').forEach(function(b) {
          b.classList.toggle('active', b.dataset.lang === cur);
        });
      }
    })();

    // Setup wizard language picker
    on('setup-language', 'change', function() {
      var lang = document.getElementById('setup-language').value;
      if (typeof I18n !== 'undefined') I18n.setLang(lang);
      var settingsLang = document.getElementById('settings-language');
      if (settingsLang) settingsLang.value = lang;
      // Toggle "Continue in English" visibility
      var engBtn = document.getElementById('setup-lang-english');
      if (engBtn) engBtn.style.display = (lang !== 'en') ? '' : 'none';
    });

    on('btn-rerun-setup', 'click', function() {
      localStorage.removeItem(SETUP_DONE_KEY);
      ui.closeSettings();
      setup.currentStep = 1;
      setup.updateUI();
      document.getElementById('setup-modal').classList.remove('hidden');
      setup.detectHardware();
    });

    on('btn-show-all-peers', 'click', function() {
      dashboard._peersExpanded = !dashboard._peersExpanded;
      dashboard.loadNetworkData();
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

    // Key source dropdown
    var keySourceSel = document.getElementById('provider-key-source');
    if (keySourceSel) {
      keySourceSel.addEventListener('change', function() {
        authFetch('/api/admin/providers', {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ key_source: this.value })
        }).then(function() {
          ui.showBanner('success', 'Key source updated to: ' + keySourceSel.value);
          settings.loadProviders();
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
    on('hamburger-btn', 'click', function() { ui.toggleSidebar(); });
    on('logo', 'click', function() { ui.switchTab('dashboard'); });
    on('btn-shutdown', 'click', function() { shutdown(); });

    // Sidebar
    on('sidebar-overlay', 'click', function() { ui.closeSidebar(); });
    on('btn-new-session', 'click', function() { chat.newSession(); if (activeTab !== 'chat') ui.switchTab('chat'); });
    on('btn-close-sidebar', 'click', function() { ui.closeSidebar(); });

    // Edge-trigger hover: pop sidebar out when hovering left edge on non-chat tabs
    // Float-mode sidebar: hover tab or sidebar body to peek; leave to collapse
    var _sidebarHoverTimer = null;
    var sidebarEl = document.getElementById('sidebar');
    if (sidebarEl) {
      sidebarEl.addEventListener('mouseenter', function() {
        clearTimeout(_sidebarHoverTimer);
        if (this.classList.contains('sidebar-float')) this.classList.remove('collapsed');
      });
      sidebarEl.addEventListener('mouseleave', function() {
        if (this.classList.contains('sidebar-float')) {
          _sidebarHoverTimer = setTimeout(function() {
            var s = document.getElementById('sidebar');
            if (s && s.classList.contains('sidebar-float')) s.classList.add('collapsed');
          }, 120);
        }
      });
    }

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
        var sidebar = document.getElementById('sidebar');
        var settingsModal = document.getElementById('settings-modal');
        var modelModal = document.getElementById('model-browser-modal');
        if (sidebar && !sidebar.classList.contains('collapsed') && window.innerWidth < 768) { ui.closeSidebar(); }
        else if (settingsModal && !settingsModal.classList.contains('hidden')) { ui.closeSettings(); }
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

      var cloudRow = target.closest('[data-select-cloud]');
      if (cloudRow) { selectModelDropdown(cloudRow.getAttribute('data-select-cloud')); chat.newSession(); ui.switchTab('chat'); return; }

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

      // Encrypted pipeline lock badge click
      var encToggle = target.getAttribute('data-enc-toggle') || (target.closest('[data-enc-toggle]') || {}).getAttribute && (target.closest('[data-enc-toggle]') || {}).getAttribute('data-enc-toggle');
      if (encToggle) {
        var encReady = (target.getAttribute('data-enc-ready') || (target.closest('[data-enc-ready]') || {}).getAttribute && (target.closest('[data-enc-ready]') || {}).getAttribute('data-enc-ready')) === '1';
        if (encReady) {
          // Toggle encrypted pipeline
          var isActive = target.classList.contains('active') || (target.closest('.active') != null);
          authFetch('/api/admin/models/' + encodeURIComponent(encToggle) + '/encrypted-pipeline', {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ enabled: !isActive }),
          }).then(function(r) {
            if (r.ok) {
              ui.showBanner('success', (!isActive ? 'Encrypted pipeline enabled' : 'Encrypted pipeline disabled') + ' for ' + encToggle);
              refreshModels();
            } else {
              ui.showBanner('error', 'Failed to toggle encrypted pipeline');
            }
          });
        } else {
          // Not ready — offer to download missing shards
          if (confirm('Encrypted pipeline requires first + last shard on this node.\n\nDownload missing shards from HuggingFace?')) {
            authFetch('/api/admin/hf/source/' + encodeURIComponent(encToggle)).then(function(r) {
              if (!r.ok) { ui.showBanner('error', 'No HuggingFace source found for ' + encToggle); return; }
              return r.json();
            }).then(function(src) {
              if (!src) return;
              // Find which endpoint shards are missing
              var modelData = (window._lastModelsData || []).find(function(mm) { return mm.id === encToggle; });
              var missing = [];
              if (modelData) {
                var first = modelData.shards[0];
                var last = modelData.shards[modelData.shards.length - 1];
                if (first && !first.local) missing.push(first.index);
                if (last && !last.local) missing.push(last.index);
              }
              if (missing.length === 0) { ui.showBanner('info', 'No missing endpoint shards detected'); return; }
              authFetch('/api/admin/hf/download-shards', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ repo_id: src.repo_id, filename: src.filename, shards: missing }),
              }).then(function(r) {
                if (r.ok) ui.showBanner('success', 'Downloading shard(s) ' + missing.join(', ') + ' for encrypted pipeline');
                else ui.showBanner('error', 'Download failed');
              });
            });
          }
        }
        return;
      }

      // Auto-manage gear icon
      var gearId = target.getAttribute('data-am-gear');
      if (gearId) { toggleAutoManagePanel(gearId); return; }

      // Auto-manage save button
      var amSave = target.getAttribute('data-am-save');
      if (amSave) { saveAutoManagePolicy(amSave); return; }

      // Model card click → select model and switch to chat
      var modelCard = target.closest('.model-card');
      if (modelCard && !target.closest('button, a, .shard-cell, .badge-encrypted, [data-cancel-download], [data-remove-model], [data-unload-model], [data-enc-toggle], [data-am-gear], input, select')) {
        var cardModelId = modelCard.getAttribute('data-model-id');
        if (cardModelId) {
          var cardModel = (window._lastModelsData || []).find(function(mm) { return mm.id === cardModelId; });
          var cardReady = cardModel && (cardModel.status === 'loaded' || cardModel.status === 'ready' ||
            (cardModel.global_available === cardModel.shard_count && cardModel.shard_count > 0));
          if (cardReady) {
            selectModelDropdown(cardModelId);
            chat.newSession();
            ui.switchTab('chat');
          } else {
            ui.showBanner('warning', 'Model not ready — download all shards first');
          }
          return;
        }
      }

      // Compare card copy button
      var copyCompare = target.getAttribute('data-copy-compare');
      if (copyCompare) {
        var el = document.getElementById(copyCompare);
        if (el) {
          navigator.clipboard.writeText(el.textContent).then(function() {
            target.textContent = 'Copied!';
            setTimeout(function() { target.textContent = 'Copy'; }, 1500);
          });
        }
        return;
      }

      // Compare history restore
      var historyRow = target.closest('[data-compare-idx]');
      if (historyRow) {
        var idx = parseInt(historyRow.getAttribute('data-compare-idx'), 10);
        try {
          var hist = JSON.parse(localStorage.getItem('swarmllm_compare_history') || '[]');
          if (hist[idx]) compare.restoreFromHistory(hist[idx]);
        } catch (e) {}
        return;
      }

      // Chat action buttons (copy, compare)
      if (target.getAttribute('data-action') === 'copy') {
        var msgEl = target.closest('.chat-msg');
        var contentEl = msgEl ? msgEl.querySelector('.msg-content') : null;
        if (contentEl) {
          navigator.clipboard.writeText(contentEl.textContent).then(function() {
            target.textContent = 'Copied!';
            setTimeout(function() { target.textContent = 'Copy'; }, 1500);
          });
        }
        return;
      }
      if (target.getAttribute('data-action') === 'compare') {
        // Find the user message that preceded this assistant message
        var msgEl = target.closest('.chat-msg');
        if (msgEl) {
          var prev = msgEl.previousElementSibling;
          while (prev && !prev.classList.contains('user')) prev = prev.previousElementSibling;
          if (prev) {
            var userContent = prev.querySelector('.msg-content');
            if (userContent) {
              // Switch to compare tab and populate prompt
              ui.switchTab('compare');
              var promptEl = document.getElementById('compare-prompt');
              if (promptEl) promptEl.value = userContent.textContent;
              compare.loadModels();
              showToast('Your question is ready \u2014 pick models and hit Compare', 'info');
            }
          }
        }
        return;
      }

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
  function formatModelDisplayName(id, opts) {
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
    var hideQuant = (opts && opts.hideQuant) || false;
    // Split on separators and format each part
    return name.split(/[-_.]/).filter(Boolean).map(function(s) {
      s = s.replace(/\x00/g, '.'); // restore decimal dots
      // Keep quant tags uppercase (Q4_K_M, Q5_K_S, etc.)
      if (/^(q\d|iq\d|f16|f32|bf16)/i.test(s)) return hideQuant ? null : s.toUpperCase();
      // Keep version strings as-is (v1, v0.3)
      if (/^v\d/i.test(s)) return s;
      // Keep size designators (1b, 7b, 1.1b)
      if (/^\d+\.?\d*[bBmM]$/.test(s)) return s.toUpperCase();
      // Strip bare 'k', 'm', 's' quant suffixes (from Q4_K_M split)
      if (hideQuant && /^[kms]$/i.test(s)) return null;
      return s.charAt(0).toUpperCase() + s.slice(1);
    }).filter(Boolean).join(' ');
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
        if (!dlInfo) chatInput.placeholder = 'No models available \u2014 click + above to download one, or add a cloud provider in Settings';
      }
    }
    if (emptyState && !hasModels) {
      emptyState.innerHTML = '<div class="chat-empty-icon">&#11203;</div>' +
        '<div class="chat-empty-title" style="font-size:1.1rem">No Models Available</div>' +
        '<div class="chat-empty-hint" style="margin:8px 0">Download an AI model to run locally, or add a cloud provider in Settings for instant access</div>' +
        '<div style="display:flex;gap:8px;margin-top:12px">' +
          '<button class="btn btn-primary" data-goto-browse="1">Download Model</button>' +
          '<button class="btn btn-outline" data-goto-network-code="1" style="border:1px solid var(--border)">' + 'Share Network Code' + '</button>' +
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

    var cloudCount = 0;
    var cloudDown = 0;
    // Count from provider list if available
    var seen = {};
    if (providerData && providerData.providers) {
      providerData.providers.forEach(function(p) {
        if (!p.configured) return;
        seen[p.name] = true;
        var h = providerHealth[p.name] || providerHealth[p.provider];
        var isHealthy = !h || h.status === 'up' || h.status === 'rate_limited' || h.status === 'overloaded';
        if (isHealthy) cloudCount++;
        else cloudDown++;
      });
    }
    // Also count from providerHealth directly (covers case where provider list fetch failed/pending)
    Object.keys(providerHealth).forEach(function(key) {
      if (seen[key]) return;
      var h = providerHealth[key];
      var isHealthy = h.status === 'up' || h.status === 'rate_limited' || h.status === 'overloaded';
      if (isHealthy) cloudCount++;
      else cloudDown++;
    });

    // Remove old mode classes
    if (indicator) indicator.className = 'mode-indicator mb-2';

    // Determine mode
    var modeName, dotClass, modeClass, modeHelp;

    if (peers > 0 && hasLocalModel && cloudCount > 0) {
      modeName = 'SWARM \u00b7 CLOUD';
      dotClass = 'swarm';
      modeClass = 'mode-hybrid';
      modeHelp = 'Full power — swarm inference with cloud fallback';
    } else if (peers > 0 && hasLocalModel) {
      modeName = 'SWARM';
      dotClass = 'swarm';
      modeClass = 'mode-swarm';
      modeHelp = 'Running inference locally and with peers';
    } else if (peers > 0) {
      modeName = 'SWARM \u00b7 REMOTE';
      dotClass = 'swarm';
      modeClass = 'mode-swarm';
      modeHelp = 'Using peer nodes for inference (no local model)';
    } else if (hasLocalModel && cloudCount > 0) {
      modeName = 'LOCAL \u00b7 CLOUD';
      dotClass = 'hybrid';
      modeClass = 'mode-hybrid';
      modeHelp = 'Local inference with cloud fallback';
    } else if (hasLocalModel) {
      modeName = 'SOLO';
      dotClass = 'offline';
      modeClass = 'mode-offline';
      modeHelp = 'Local inference only — connect peers to unlock bigger models';
    } else if (cloudCount > 0) {
      modeName = 'CLOUD';
      dotClass = 'cloud';
      modeClass = 'mode-cloud';
      modeHelp = 'Using cloud providers — download models for free local AI';
    } else {
      modeName = 'OFFLINE';
      dotClass = 'offline';
      modeClass = 'mode-offline';
      modeHelp = 'Download a model or add a cloud provider to get started';
    }

    dot.className = 'mode-dot ' + dotClass;
    label.textContent = modeName;
    label.title = modeHelp;
    if (indicator) indicator.classList.add(modeClass);

    // Right side: live stats
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
      detailHtml = '<span class="mode-action" data-goto-hf="1">Get started \u2014 download a model or add a provider</span>';
    }

    detail.innerHTML = detailHtml;
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
      var resp2 = await authFetch('/api/admin/providers');
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
    // Initialize i18n — translates static [data-i18n] elements
    if (typeof I18n !== 'undefined') {
      I18n.init(['en','es','fr','de','pt','it','nl','ru','zh','ja','ko','ar','tr','pl','sv','th','hi','vi','id','uk','cs']);
    }

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

    // Load API key BEFORE switching tabs (compare tab needs auth)
    setup.init();
    settings.init();
    settings._apiKeyPromise = settings.loadApiKey();

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

    applyTheme(localStorage.getItem(THEME_KEY) || 'dark');

    // Sync setup language dropdown with detected language
    if (typeof I18n !== 'undefined') {
      var detectedLang = I18n.getLang() || 'en';
      var setupLang = document.getElementById('setup-language');
      if (setupLang) setupLang.value = detectedLang;
      // Show "Continue in English" button if non-English was auto-detected
      var engBtn = document.getElementById('setup-lang-english');
      if (engBtn && detectedLang !== 'en') {
        engBtn.style.display = '';
        engBtn.addEventListener('click', function() {
          I18n.setLang('en');
          if (setupLang) setupLang.value = 'en';
          var settingsLang = document.getElementById('settings-language');
          if (settingsLang) settingsLang.value = 'en';
          engBtn.style.display = 'none';
        });
      }
    }

    // Initialize neural network background
    if (typeof NeuralBg !== 'undefined') NeuralBg.init();

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

  // Delegated error handler for provider icons — replaces inline onerror (blocked by CSP)
  document.addEventListener('error', function(e) {
    var t = e.target;
    if (t.tagName !== 'IMG' || !t.classList.contains('provider-icon')) return;
    if (t.classList.contains('provider-avatar-icon')) {
      // Avatar fallback: replace broken img with text initials
      var av = t.parentNode;
      if (av) av.textContent = 'AI';
    } else {
      t.style.display = 'none';
    }
  }, true); // capture phase to catch before bubbling

  // Start when DOM is ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

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
          if (resp.ok) {
            var d = await resp.json();
            localModels = Array.isArray(d) ? d : (d.models || d.data || []);
          }
        } catch(e) {}
        try {
          var resp2 = await authFetch('/api/admin/provider-models');
          if (resp2.ok) {
            var d2 = await resp2.json();
            cloudModels = Array.isArray(d2) ? d2 : (d2.models || d2.data || []);
          }
        } catch(e) {}

        compare.models = [];
        (localModels || []).forEach(function(m) {
          compare.models.push({ id: m.id || m.model_id || m.name, type: 'local' });
        });
        (cloudModels || []).forEach(function(m) {
          var mid = m.id || m.model_id || m.name;
          // Deduplicate
          if (!compare.models.some(function(x) { return x.id === mid; })) {
            var ctx = m.context_length || m.context_window || m.max_model_len || 0;
            compare.models.push({ id: mid, type: 'cloud', context: ctx });
          }
        });

        if (compare.models.length === 0) {
          container.innerHTML = '<span class="text-muted" style="font-size:0.8rem">No models available yet. Download a model or add a cloud provider in Settings first.</span>';
          return;
        }

        container.innerHTML = '';
        compare.models.forEach(function(m, idx) {
          var chip = document.createElement('label');
          chip.className = 'compare-model-chip type-' + m.type;
          chip.style.animationDelay = (idx * 30) + 'ms';
          var displayName = m.id.length > 35 ? m.id.substring(0, 35) + '...' : m.id;
          var ctxLabel = m.context && m.context > 0 ? ' \u00B7 ' + Math.round(m.context / 1000) + 'k ctx' : '';
          chip.innerHTML = '<input type="checkbox" value="' + escapeHtml(m.id) + '">' +
            '<span>' + escapeHtml(displayName) + '</span>' +
            '<span class="chip-type">' + m.type + ctxLabel + '</span>';
          chip.title = m.id + (ctxLabel ? ' (' + m.context + ' tokens)' : '');
          chip.querySelector('input').addEventListener('change', function() {
            chip.classList.toggle('selected', this.checked);
            compare.updateSelected();
          });
          container.appendChild(chip);
        });

        // Wire up filter buttons
        var filters = document.getElementById('compare-filters');
        if (filters) {
          filters.querySelectorAll('.compare-filter').forEach(function(btn) {
            btn.addEventListener('click', function() {
              filters.querySelectorAll('.compare-filter').forEach(function(b) { b.classList.remove('active'); });
              btn.classList.add('active');
              var f = btn.getAttribute('data-filter');
              container.querySelectorAll('.compare-model-chip').forEach(function(chip) {
                if (f === 'all') { chip.style.display = ''; }
                else { chip.style.display = chip.classList.contains('type-' + f) ? '' : 'none'; }
              });
            });
          });
        }
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
        var controller = new AbortController();
        var timeoutId = setTimeout(function() { controller.abort(); }, 45000);
        return authFetch('/v1/messages', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body),
          signal: controller.signal,
        }).then(function(resp) {
          clearTimeout(timeoutId);
          var elapsed = Math.round(performance.now() - start);
          return resp.json().then(function(data) {
            return { model: modelId, data: data, ok: resp.ok, latency_ms: elapsed };
          });
        }).catch(function(err) {
          clearTimeout(timeoutId);
          var msg = err.name === 'AbortError' ? 'Timed out after 45s' : err.message;
          return { model: modelId, error: msg, ok: false, latency_ms: Math.round(performance.now() - start) };
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

      // Wait for all to finish, save to history
      Promise.all(promises).then(function(results) {
        compare.running = false;
        if (btn) { btn.disabled = false; btn.textContent = 'Run Compare'; }
        // Save to compare history (keep last 20)
        try {
          var history = JSON.parse(localStorage.getItem('swarmllm_compare_history') || '[]');
          history.unshift({
            prompt: prompt.trim().substring(0, 200),
            models: compare.selected.slice(),
            timestamp: Date.now(),
            results: results.map(function(r) {
              var content = '';
              if (!r.error && r.ok) {
                (r.data.content || []).forEach(function(b) { if (b.type === 'text') content += b.text; });
              }
              return {
                model: r.model, ok: r.ok, error: r.error || null,
                latency_ms: r.latency_ms, content: content,
                input_tokens: r.ok ? ((r.data.usage || {}).input_tokens || 0) : 0,
                output_tokens: r.ok ? ((r.data.usage || {}).output_tokens || 0) : 0,
              };
            }),
          });
          if (history.length > 20) history = history.slice(0, 20);
          localStorage.setItem('swarmllm_compare_history', JSON.stringify(history));
          compare.renderHistory();
        } catch (e) {}
      });
    },

    renderHistory: function() {
      var container = document.getElementById('compare-history');
      if (!container) return;
      try {
        var history = JSON.parse(localStorage.getItem('swarmllm_compare_history') || '[]');
        if (history.length === 0) { container.style.display = 'none'; return; }
        container.style.display = '';
        var html = '<div style="font-size:0.75rem;color:var(--text-muted);margin-bottom:8px;text-transform:uppercase;letter-spacing:0.06em">Recent Comparisons</div>';
        history.slice(0, 10).forEach(function(item, idx) {
          var ago = compare.timeAgo(item.timestamp);
          var modelList = (item.models || []).map(function(m) {
            return m.split('/').pop().replace(/-\d{4}-\d{2}-\d{2}$/, '');
          }).join(', ');
          html += '<div class="compare-history-item" data-compare-idx="' + idx + '">' +
            '<span class="compare-history-prompt">' + escapeHtml(item.prompt) + '</span>' +
            '<span class="compare-history-meta">' + escapeHtml(modelList) + ' &middot; ' + ago + '</span>' +
          '</div>';
        });
        container.innerHTML = html;
      } catch (e) { container.style.display = 'none'; }
    },

    restoreFromHistory: function(item) {
      var promptEl = document.getElementById('compare-prompt');
      if (promptEl) promptEl.value = item.prompt;

      var resultsDiv = document.getElementById('compare-results');
      if (!resultsDiv || !item.results || !item.results.length) return;

      resultsDiv.innerHTML = '';
      item.results.forEach(function(r) {
        var card = document.createElement('div');
        card.className = 'compare-card';
        card.id = 'compare-card-' + r.model.replace(/[^a-zA-Z0-9_-]/g, '_');
        card.innerHTML = '<div class="compare-card-body"></div>';
        resultsDiv.appendChild(card);
        compare.renderCard({
          model: r.model, ok: r.ok, error: r.error,
          latency_ms: r.latency_ms,
          data: {
            content: [{ type: 'text', text: r.content || '' }],
            usage: { input_tokens: r.input_tokens, output_tokens: r.output_tokens },
          },
        });
      });

      var statusDiv = document.getElementById('compare-status');
      if (statusDiv) { statusDiv.style.display = ''; statusDiv.innerHTML = '<span class="text-muted">Restored from history &middot; ' + compare.timeAgo(item.timestamp) + '</span>'; }
    },

    timeAgo: function(ts) {
      var s = Math.floor((Date.now() - ts) / 1000);
      if (s < 60) return 'just now';
      if (s < 3600) return Math.floor(s / 60) + 'm ago';
      if (s < 86400) return Math.floor(s / 3600) + 'h ago';
      return Math.floor(s / 86400) + 'd ago';
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

      var cardContentId = 'compare-content-' + result.model.replace(/[^a-zA-Z0-9_-]/g, '_');
      card.innerHTML =
        '<div class="compare-card-header">' +
          '<div style="display:flex;align-items:center;gap:8px;flex:1;min-width:0">' +
            '<span class="compare-card-model" style="overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="' + escapeHtml(result.model) + '">' + escapeHtml(result.model) + '</span>' +
            (isError ? '<span style="color:var(--red);font-size:0.7rem">error</span>' : '<span style="color:var(--green);font-size:0.7rem">' + result.latency_ms + 'ms</span>') +
          '</div>' +
          '<div class="compare-card-actions">' +
            '<button data-copy-compare="' + cardContentId + '" title="Copy response">Copy</button>' +
          '</div>' +
        '</div>' +
        '<div class="compare-card-body' + (isError ? ' error' : '') + '" id="' + cardContentId + '">' + escapeHtml(content) + '</div>' +
        (isError ? '' :
          '<div class="compare-card-footer">' +
            '<span>In: ' + inputTokens + '</span>' +
            '<span>Out: ' + outputTokens + '</span>' +
            '<span>' + result.latency_ms + 'ms</span>' +
            (outputTokens > 0 ? '<span>' + (function() { var t = outputTokens / (result.latency_ms / 1000); return t >= 1 ? Math.round(t) : t.toFixed(1); })() + ' tok/s</span>' : '') +
          '</div>'
        );
    },
  };

  // Post-init: load compare models if that tab is active on page load.
  // compare is defined after init(), so switchTab('compare') during init()
  // can't reach compare.loadModels(). Defer to ensure init() has completed.
  setTimeout(function() {
    if (activeTab === 'compare' && compare) {
      compare.loadModels();
      compare.renderHistory();
    }
  }, 0);

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
