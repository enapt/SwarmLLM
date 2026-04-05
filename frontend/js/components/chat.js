'use strict';

// ============================================================================
// SwarmLLM — Chat Component
// Sessions, messages, streaming, image upload, layout toggle
// ============================================================================

(function() {
  var S = App.state;
  var U = App.utils;

  // --- Image Upload ---
  function addPendingImage(file) {
    if (!file.type.startsWith('image/')) return;
    if (S.pendingImages.length >= 4) {
      App.ui.showBanner('warning', I18n.t('chat.max_images'));
      return;
    }
    var reader = new FileReader();
    reader.onload = function(e) {
      S.pendingImages.push({ data_url: e.target.result, name: file.name });
      renderImagePreviews();
    };
    reader.readAsDataURL(file);
  }

  function renderImagePreviews() {
    var area = document.getElementById('image-preview-area');
    if (!area) return;
    if (S.pendingImages.length === 0) {
      area.style.display = 'none';
      area.innerHTML = '';
      return;
    }
    area.style.display = 'flex';
    area.style.flexWrap = 'wrap';
    area.style.gap = '6px';
    area.innerHTML = '';
    S.pendingImages.forEach(function(img, idx) {
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
        S.pendingImages.splice(idx, 1);
        renderImagePreviews();
      };
      wrap.appendChild(thumb);
      wrap.appendChild(removeBtn);
      area.appendChild(wrap);
    });
  }

  function clearPendingImages() {
    S.pendingImages = [];
    renderImagePreviews();
  }

  function buildMessageContent(text, images) {
    if (!images || images.length === 0) return text;
    var parts = [];
    images.forEach(function(img) {
      parts.push({ type: 'image_url', image_url: { url: img.data_url } });
    });
    parts.push({ type: 'text', text: text || I18n.t('chat.default_image_prompt') });
    return parts;
  }

  // --- Chat Layout Toggle ---
  function toggleChatLayout() {
    var container = document.getElementById('chat-messages');
    var btn = document.getElementById('chat-layout-toggle');
    var icon = document.getElementById('chat-layout-icon');
    var label = document.getElementById('chat-layout-label');
    if (!container) return;
    var isMessenger = container.classList.toggle('chat-messenger');
    if (icon) icon.innerHTML = isMessenger ? '&#9900;' : '&#9776;';
    if (label) label.textContent = isMessenger ? I18n.t('chat.layout_messenger') : I18n.t('chat.layout_linear');
    if (btn) btn.classList.toggle('active', isMessenger);
    try { localStorage.setItem(App.CHAT_LAYOUT_KEY, isMessenger ? 'messenger' : 'linear'); } catch(e) {}
    App.chat.scrollToBottom();
  }

  function initChatLayout() {
    try {
      var saved = localStorage.getItem(App.CHAT_LAYOUT_KEY);
      if (saved === 'messenger') {
        var container = document.getElementById('chat-messages');
        var icon = document.getElementById('chat-layout-icon');
        var label = document.getElementById('chat-layout-label');
        var btn = document.getElementById('chat-layout-toggle');
        if (container) container.classList.add('chat-messenger');
        if (icon) icon.innerHTML = '&#9900;';
        if (label) label.textContent = I18n.t('chat.layout_messenger');
        if (btn) btn.classList.add('active');
      }
    } catch(e) {}
  }

  // --- Chat Module ---
  App.chat = {
    // Expose for init
    addPendingImage: addPendingImage,
    clearPendingImages: clearPendingImages,
    buildMessageContent: buildMessageContent,
    toggleChatLayout: toggleChatLayout,
    initChatLayout: initChatLayout,

    handleKey: function(e) {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        App.chat.send();
      }
    },

    newSession: function() {
      if (S.currentSessionId && S.sessions[S.currentSessionId] && S.sessions[S.currentSessionId].messages.length === 0) {
        S.sessions[S.currentSessionId].model = S.currentModel || '';
        App.chat.saveSessions();
        App.chat.renderSessionList();
        App.chat.renderMessages();
        App.chat.updateChatHeader();
        return;
      }
      var emptied = [];
      Object.keys(S.sessions).forEach(function(sid) {
        if (S.sessions[sid].messages.length === 0 && sid !== S.currentSessionId) {
          emptied.push(sid);
          delete S.sessions[sid];
        }
      });
      var id = 'session_' + Date.now();
      S.sessions[id] = { id: id, title: I18n.t('chat.new_chat'), messages: [], created: Date.now(), model: S.currentModel || '' };
      S.currentSessionId = id;
      App.chat.saveSessions();
      App.chat.renderSessionList();
      App.chat.renderMessages();
      App.chat.updateChatHeader();
      if (emptied.length > 0) {
        App.notifications.showToast(I18n.t('chat.cleaned_sessions', { count: emptied.length }), 'info', 3000);
      }
      App.ui.switchTab('chat');
    },

    switchSession: function(id) {
      if (!S.sessions[id]) return;
      S.currentSessionId = id;
      localStorage.setItem(App.ACTIVE_SESSION_KEY, id);

      var s = S.sessions[id];
      if (s.model) {
        var allIds = S._modelDropdownData.map(function(m) { return m.id; });
        if (allIds.indexOf(s.model) !== -1) {
          App.models.selectDropdown(s.model, { silent: true });
        } else if (s.messages.length > 0) {
          App.notifications.showToast(I18n.t('chat.model_unavailable_readonly', { model: U.formatModelDisplayName(s.model) }), 'warning');
        }
      }

      App.chat.renderSessionList();
      App.chat.renderMessages();
      App.chat.updateChatHeader();
      if (window.innerWidth < 768) App.ui.closeSidebar();
    },

    deleteSession: function(id, e) {
      if (e) { e.stopPropagation(); e.preventDefault(); }
      delete S.sessions[id];
      if (S.currentSessionId === id) {
        var keys = Object.keys(S.sessions);
        S.currentSessionId = keys.length > 0 ? keys[keys.length - 1] : null;
      }
      App.chat.saveSessions();
      App.chat.renderSessionList();
      App.chat.renderMessages();
    },

    renderSessionList: function() {
      var list = document.getElementById('session-list');
      if (!list) return;
      var sorted = Object.values(S.sessions).sort(function(a, b) { return b.created - a.created; });
      if (sorted.length === 0) {
        list.innerHTML = '<div class="text-muted" style="padding:12px;font-size:0.8rem">' + U.escapeHtml(I18n.t('chat.no_chats_yet')) + '</div>';
        return;
      }
      list.innerHTML = '';
      var tmpl = document.getElementById('tmpl-session-item');
      sorted.forEach(function(s) {
        var div = tmpl.content.cloneNode(true).firstElementChild;
        if (s.id === S.currentSessionId) div.classList.add('active');
        div.onclick = function() { App.chat.switchSession(s.id); if (S.activeTab !== 'chat') App.ui.switchTab('chat'); };

        // Title
        var titleEl = div.querySelector('.session-title');
        var title = s.title.length > 28 ? s.title.substring(0, 28) + '...' : s.title;
        titleEl.textContent = title;
        titleEl.setAttribute('data-rename-session', s.id);

        // Time
        var timeEl = div.querySelector('.session-time');
        if (s.created) {
          timeEl.textContent = new Date(s.created).toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' });
        }

        // Model badge
        var modelItem = s.model ? S._modelDropdownData.find(function(m) { return m.id === s.model; }) : null;
        var badgeEl = div.querySelector('.session-model-badge');
        if (s.model) {
          var source = U.getModelSource(s.model);
          var sourceLabel = source === 'local' ? I18n.t('chat.source_local') : source === 'cloud' ? I18n.t('chat.source_cloud') : I18n.t('chat.source_network');
          var _sibIconKey = (modelItem && modelItem.group && _ICON_MAP[modelItem.group]) ? modelItem.group : modelIconKey(s.model);
          var sibIconHtml = _sibIconKey ? providerIconHtml(_sibIconKey, 11) : '';
          badgeEl.removeAttribute('hidden');
          badgeEl.className = 'session-model-badge session-source-' + source;
          var tooltipParts = [s.model];
          if (source !== 'local') tooltipParts.push(sourceLabel);
          if (modelItem && modelItem.encrypted) tooltipParts.push(I18n.t('chat.enc_pipeline_tooltip'));
          badgeEl.title = tooltipParts.join(' \u2022 ');
          badgeEl.innerHTML = (sibIconHtml ? sibIconHtml + ' ' : '') + U.escapeHtml(U.formatModelDisplayName(s.model));
        }

        // Encryption badge
        if (modelItem && modelItem.encrypted) {
          div.querySelector('.session-enc-lock').removeAttribute('hidden');
        }

        // Claude Code badge
        if (App.claudeCode) {
          var ccBadge = App.claudeCode.getSessionBadge(s);
          if (ccBadge) {
            var ccEl = document.createElement('span');
            ccEl.className = 'session-cc-badge';
            ccEl.title = ccBadge.dir;
            ccEl.textContent = ccBadge.dir;
            var stateClass = ccBadge.state === 'active' ? 'cc-active' : ccBadge.state === 'suspended' ? 'cc-suspended' : '';
            if (stateClass) ccEl.classList.add(stateClass);
            var titleRow = div.querySelector('.session-title');
            if (titleRow && titleRow.parentNode) titleRow.parentNode.insertBefore(ccEl, titleRow.nextSibling);
          }
        }

        // Delete button
        div.querySelector('.session-delete').setAttribute('data-delete-session', s.id);

        list.appendChild(div);
      });
    },

    renameSession: function(id, titleEl) {
      if (!S.sessions[id]) return;
      var current = S.sessions[id].title;
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
          S.sessions[id].title = val;
          App.chat.saveSessions();
        }
        App.chat.renderSessionList();
        App.chat.updateChatHeader();
      };
      input.addEventListener('blur', done);
      input.addEventListener('keydown', function(e) {
        if (e.key === 'Enter') { e.preventDefault(); input.blur(); }
        if (e.key === 'Escape') { input.value = current; input.blur(); }
      });
    },

    updateChatHeader: function() {
      var header = document.getElementById('chat-session-header');
      var encBanner = document.getElementById('chat-enc-banner');
      if (!header) return;
      // Update Claude Code project bar
      if (App.claudeCode) App.claudeCode.updateProjectBar();
      if (!S.currentSessionId || !S.sessions[S.currentSessionId]) {
        header.classList.remove('visible');
        header.innerHTML = '';
        if (encBanner) encBanner.style.display = 'none';
        return;
      }
      var s = S.sessions[S.currentSessionId];
      var modelName = s.model ? U.formatModelDisplayName(s.model) : I18n.t('chat.no_model');
      var allIds = S._modelDropdownData.map(function(m) { return m.id; });
      var available = !s.model || allIds.indexOf(s.model) !== -1;
      var headerSource = U.getModelSource(s.model || '');
      var badgeClass = 'chat-session-model source-' + headerSource + (available ? '' : ' unavailable');
      var badgeTitle = available ? s.model : I18n.t('chat.model_unavailable');
      var headerModelItem = s.model ? S._modelDropdownData.find(function(m) { return m.id === s.model; }) : null;
      var isEncrypted = headerModelItem && headerModelItem.encrypted;
      var _hdrIconKey = (headerModelItem && headerModelItem.group && _ICON_MAP[headerModelItem.group]) ? headerModelItem.group : modelIconKey(s.model || '');
      var hdrIconHtml = _hdrIconKey ? providerIconHtml(_hdrIconKey, 12) : '';
      var msgCount = s.messages.length;
      var countLabel = msgCount === 0 ? I18n.t('chat.count_new') : (msgCount === 1 ? I18n.t('chat.count_one') : I18n.t('chat.count_many', { count: msgCount }));
      var countClass = 'chat-session-count' + (msgCount === 0 ? ' is-new' : '');
      var safeModelId = U.escapeHtml(s.model || '');
      // Subscription/API badge for the model
      var authBadge = '';
      if (headerModelItem && headerModelItem.group === 'claude_subscription') {
        authBadge = ' <span class="cc-auth-badge cc-auth-sub" title="' + U.escapeHtml(I18n.t('claude_code.subscription_tip')) + '">Sub</span>';
      } else if (headerSource === 'cloud') {
        authBadge = ' <span class="cc-auth-badge cc-auth-api">API</span>';
      }

      header.classList.add('visible');
      header.innerHTML =
        '<span class="chat-session-title" id="chat-header-title" title="' + U.escapeHtml(I18n.t('chat.rename_title')) + '">' + U.escapeHtml(s.title) + '</span>' +
        '<span class="' + countClass + '">' + U.escapeHtml(countLabel) + '</span>' +
        '<span class="' + badgeClass + '" title="' + U.escapeHtml(badgeTitle) + '">' + (hdrIconHtml ? hdrIconHtml + ' ' : '') + U.escapeHtml(modelName) + authBadge + (available ? '' : ' ' + U.escapeHtml(I18n.t('chat.model_unavailable_suffix'))) + '</span>';

      if (encBanner) {
        var modelData = s.model ? (App.data.cache.models || []).find(function(m) { return m.id === s.model; }) : null;
        var isDistributed = modelData && modelData.shard_count > 1;
        var isAllLocal = modelData && modelData.hosted_shards === modelData.shard_count && modelData.shard_count > 0;
        var canBoomerang = modelData && modelData.has_first_shard && modelData.has_last_shard && isDistributed && !isAllLocal;
        var disableBtn = '<button class="btn btn-xs enc-banner-btn" data-enc-toggle="' + safeModelId + '" data-enc-ready="1">' + U.escapeHtml(I18n.t('enc.disable')) + '</button>';
        var enableBtn = '<button class="btn btn-xs enc-banner-btn enc-banner-btn-enable" data-enc-toggle="' + safeModelId + '" data-enc-ready="1">' + U.escapeHtml(I18n.t('enc.enable_privacy')) + '</button>';
        if (headerSource === 'cloud') {
          var providerName = (headerModelItem && headerModelItem.group) ? (PROVIDER_NAMES[headerModelItem.group] || headerModelItem.group) : I18n.t('chat.unknown_provider');
          var providerIcon = (headerModelItem && headerModelItem.group) ? providerIconHtml(headerModelItem.group, 12) : '';
          encBanner.className = 'chat-enc-banner enc-cloud';
          encBanner.innerHTML = (providerIcon ? providerIcon + ' ' : '') +
            U.escapeHtml(I18n.t('chat.cloud_routing', { provider: providerName }));
          encBanner.style.display = '';
        } else if (isAllLocal) {
          encBanner.className = 'chat-enc-banner enc-local';
          encBanner.innerHTML = '&#128187; ' + U.escapeHtml(I18n.t('enc.running_locally'));
          encBanner.style.display = '';
        } else if (isDistributed && isEncrypted) {
          encBanner.className = 'chat-enc-banner enc-boomerang';
          encBanner.innerHTML = '&#128274; ' + U.escapeHtml(I18n.t('enc.full_e2e')) +
            ' \u00b7 <span class="enc-overhead">' + U.escapeHtml(I18n.t('enc.full_e2e_overhead')) + '</span> ' + disableBtn;
          encBanner.style.display = '';
        } else if (isDistributed) {
          encBanner.className = 'chat-enc-banner enc-warn';
          encBanner.innerHTML = '&#128275; ' + U.escapeHtml(I18n.t('enc.transport_encrypted')) +
            ' \u00b7 <span class="enc-overhead">' + U.escapeHtml(I18n.t('enc.gain_speed')) + '</span>' +
            (canBoomerang ? ' ' + enableBtn : '');
          encBanner.style.display = '';
        } else {
          encBanner.style.display = 'none';
        }
      }

      var sendBtn = document.getElementById('send-btn');
      if (sendBtn) {
        var modelData2 = s.model ? (App.data.cache.models || []).find(function(m) { return m.id === s.model; }) : null;
        var sendEncActive = !!(modelData2 && modelData2.encrypted_pipeline && modelData2.shard_count > 1);
        sendBtn.innerHTML = sendEncActive
          ? App.utils.escapeHtml(I18n.t('chat.send')) + ' <span class="send-enc-lock" aria-hidden="true">&#128274;</span>'
          : App.utils.escapeHtml(I18n.t('chat.send'));
      }
    },

    renderMessages: function() {
      var container = document.getElementById('chat-messages');
      if (!container) return;
      container.innerHTML = '';

      App.chat.updateChatHeader();

      if (!S.currentSessionId || !S.sessions[S.currentSessionId]) {
        container.appendChild(U.createEmptyState());
        return;
      }

      var msgs = S.sessions[S.currentSessionId].messages;
      if (msgs.length === 0) {
        container.appendChild(U.createEmptyState());
        return;
      }

      msgs.forEach(function(msg) {
        var msgOpts = { encrypted: !!msg.encrypted };
        if (msg.images && msg.images.length > 0) {
          var html = '<div style="margin-bottom:6px;">';
          msg.images.forEach(function(url) {
            html += '<img src="' + U.escapeHtml(url) + '" style="max-height:120px;max-width:200px;border-radius:8px;margin-right:4px;" />';
          });
          html += '</div>' + U.escapeHtml(msg.content);
          U.appendMessageToDOM(msg.role, html, true, msgOpts);
        } else {
          U.appendMessageToDOM(msg.role, msg.content, false, msgOpts);
        }
      });
      App.chat.scrollToBottom();
    },

    send: async function() {
      if (S.isStreaming) return;
      if (!S.currentModel) {
        App.ui.showBanner('warning', I18n.t('chat.no_model_warning'));
        return;
      }

      if (S.currentSessionId && S.sessions[S.currentSessionId] && S.sessions[S.currentSessionId].model) {
        var allIds = S._modelDropdownData.map(function(m) { return m.id; });
        if (allIds.indexOf(S.sessions[S.currentSessionId].model) === -1) {
          App.ui.showBanner('warning', I18n.t('chat.model_unavailable_new', { model: U.formatModelDisplayName(S.sessions[S.currentSessionId].model) }));
          return;
        }
      }

      var input = document.getElementById('chat-input');
      var text = input.value.trim();
      var images = S.pendingImages.slice();
      if (!text && images.length === 0) return;

      if (!S.currentSessionId || !S.sessions[S.currentSessionId]) {
        App.chat.newSession();
      }

      input.value = '';
      U.autoResizeInput();
      clearPendingImages();

      var session = S.sessions[S.currentSessionId];
      var displayText = text || (images.length > 0 ? I18n.t('chat.image_placeholder') : '');
      var _sendModel = session.model || S.currentModel || '';
      var _sendModelData = _sendModel ? (App.data.cache.models || []).find(function(m) { return m.id === _sendModel; }) : null;
      var msgEncrypted = !!(_sendModelData && _sendModelData.encrypted_pipeline && _sendModelData.shard_count > 1);
      session.messages.push({ role: 'user', content: displayText, images: images.map(function(i) { return i.data_url; }), encrypted: msgEncrypted });

      if (session.messages.length === 1) {
        session.title = displayText.substring(0, 50);
        App.chat.renderSessionList();
      }

      App.chat.saveSessions();
      var userHtml = '';
      if (images.length > 0) {
        userHtml += '<div style="margin-bottom:6px;">';
        images.forEach(function(img) {
          userHtml += '<img src="' + U.escapeHtml(img.data_url) + '" style="max-height:120px;max-width:200px;border-radius:8px;margin-right:4px;" />';
        });
        userHtml += '</div>';
      }
      userHtml += U.escapeHtml(displayText);
      U.appendMessageToDOM('user', userHtml, true, { encrypted: msgEncrypted });

      var assistantEl = U.appendMessageToDOM('assistant', '', false, { encrypted: msgEncrypted });
      var contentEl = assistantEl.querySelector('.msg-content');
      contentEl.innerHTML = '<span class="typing-indicator">' + U.escapeHtml(I18n.t('chat.thinking')) + '</span>';

      S.isStreaming = true;
      var _sendBtn = document.getElementById('send-btn');
      if (_sendBtn) _sendBtn.disabled = true;
      var startTime = performance.now();

      var model = session.model || S.currentModel || 'local';
      if (!session.model) {
        session.model = model;
        App.chat.updateChatHeader();
        App.chat.renderSessionList();
      }

      // ── Claude Code session path ──
      if (App.claudeCode && App.claudeCode.isClaudeCodeModel(model)) {
        try {
          var cc = App.claudeCode.getSessionCC(session);
          // Create or resume backend session if not active
          if (!cc.active) {
            var dirInput = document.getElementById('cc-dir-input');
            var permSelect = document.getElementById('cc-permission-mode');
            var workDir = cc.working_dir || (dirInput ? dirInput.value.trim() : '');
            var permMode = cc.permission_mode || (permSelect ? permSelect.value : 'bypassPermissions');
            cc.permission_mode = permMode;
            // Show init status while CLI boots (hooks can take 5-15s)
            contentEl.innerHTML = '<span class="typing-indicator">' +
              U.escapeHtml(I18n.t('claude_code.initializing')) + '</span>';
            await App.claudeCode.createSession(session.id, model, workDir, permMode);
            App.claudeCode.updateProjectBar();
          }
          // Translate slash commands
          var ccText = displayText;
          var translated = App.claudeCode.translateSlashCommand(displayText);
          if (translated) ccText = translated;
          var result = await App.claudeCode.sendMessage(session.id, ccText, contentEl, assistantEl);
          if (result.content) {
            session.messages.push({ role: 'assistant', content: result.content, encrypted: false });
            App.chat.saveSessions();
          }
        } catch (e) {
          contentEl.textContent = e.message || I18n.t('chat.connection_failed');
          contentEl.classList.add('chat-error');
        }
        S.isStreaming = false;
        var _sendBtnCC = document.getElementById('send-btn');
        if (_sendBtnCC) _sendBtnCC.disabled = false;
        return;
      }

      // ── Standard OpenAI chat completions path ──
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
        var resp = await App.authFetch('/v1/chat/completions', {
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
              if (errJson.error.hint) hintHtml = '<div class="chat-error-hint">' + U.escapeHtml(errJson.error.hint) + '</div>';
            }
          } catch (e) {}
          contentEl.innerHTML = U.escapeHtml(friendlyMsg) + hintHtml + '<div class="chat-error-actions"><button class="btn btn-sm" data-retry-chat="1">' + U.escapeHtml(I18n.t('actions.retry')) + '</button></div>';
          contentEl.classList.add('chat-error');
          S.isStreaming = false;
          var _sb = document.getElementById('send-btn');
          if (_sb) _sb.disabled = false;
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
                if (delta.reasoning_content) {
                  if (!cleared) { contentEl.textContent = ''; cleared = true; }
                  if (!thinkingEl) {
                    thinkingEl = document.createElement('details');
                    thinkingEl.className = 'reasoning-block';
                    thinkingEl.innerHTML = '<summary>' + U.escapeHtml(I18n.t('chat.reasoning_label')) + '</summary><pre class="reasoning-content"></pre>';
                    thinkingEl.open = true;
                    contentEl.appendChild(thinkingEl);
                  }
                  reasoningContent += delta.reasoning_content;
                  thinkingEl.querySelector('.reasoning-content').textContent = reasoningContent;
                  App.chat.scrollToBottom();
                }
                if (delta.content) {
                  if (!cleared) { contentEl.textContent = ''; cleared = true; }
                  if (thinkingEl && thinkingEl.open) {
                    thinkingEl.open = false;
                    thinkingEl.querySelector('summary').textContent = I18n.t('chat.reasoning_summary', { chars: reasoningContent.length });
                  }
                  fullContent += delta.content;
                  var textNode = contentEl.querySelector('.response-text');
                  if (!textNode) {
                    textNode = document.createElement('div');
                    textNode.className = 'response-text';
                    contentEl.appendChild(textNode);
                  }
                  textNode.textContent = fullContent;
                  App.chat.scrollToBottom();
                }
              }
            } catch (e) {}
          }
        }

        if (!cleared && !fullContent && !reasoningContent) {
          contentEl.textContent = I18n.t('chat.no_response');
          contentEl.classList.add('chat-error');
        }
      } catch (e) {
        if (!fullContent) {
          contentEl.textContent = I18n.t('chat.connection_failed');
          contentEl.classList.add('chat-error');
        }
      }

      var elapsed = ((performance.now() - startTime) / 1000).toFixed(2);
      var timerEl = document.createElement('div');
      timerEl.className = 'msg-timer';
      timerEl.textContent = I18n.t('chat.response_time', { seconds: elapsed });
      var timerTarget = assistantEl.querySelector('.msg-bubble') || assistantEl;
      timerTarget.appendChild(timerEl);

      if (fullContent) {
        session.messages.push({ role: 'assistant', content: fullContent, encrypted: msgEncrypted });
        App.chat.saveSessions();
      }

      S.isStreaming = false;
      var _sendBtnEnd = document.getElementById('send-btn');
      if (_sendBtnEnd) _sendBtnEnd.disabled = false;
    },

    scrollToBottom: function() {
      var container = document.getElementById('chat-messages');
      if (container) container.scrollTop = container.scrollHeight;
    },

    saveSessions: function() {
      try {
        var stripped = {};
        Object.keys(S.sessions).forEach(function(id) {
          var s = S.sessions[id];
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
        localStorage.setItem(App.SESSIONS_KEY, JSON.stringify(stripped));
        if (S.currentSessionId) localStorage.setItem(App.ACTIVE_SESSION_KEY, S.currentSessionId);
      } catch (e) {}
    },

    loadSessions: function() {
      try {
        var saved = localStorage.getItem(App.SESSIONS_KEY);
        if (saved) S.sessions = JSON.parse(saved);

        var oldHistory = localStorage.getItem(App.CHAT_HISTORY_KEY);
        if (oldHistory && Object.keys(S.sessions).length === 0) {
          var msgs = JSON.parse(oldHistory);
          if (msgs.length > 0) {
            var id = 'session_migrated';
            S.sessions[id] = { id: id, title: msgs[0].content.substring(0, 50), messages: msgs, created: Date.now() - 1000 };
            localStorage.removeItem(App.CHAT_HISTORY_KEY);
          }
        }

        S.currentSessionId = localStorage.getItem(App.ACTIVE_SESSION_KEY);
        if (S.currentSessionId && !S.sessions[S.currentSessionId]) {
          S.currentSessionId = Object.keys(S.sessions).pop() || null;
        }
      } catch (e) {
        S.sessions = {};
      }
    }
  };
})();
