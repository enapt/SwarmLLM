'use strict';

// ============================================================================
// SwarmLLM — Utility Functions
// Format helpers, DOM builders, shared pure functions
// ============================================================================

(function() {
  var S = App.state;

  function escapeHtml(str) {
    // SEC: Must escape quotes for safe use in HTML attribute values.
    // The textContent/innerHTML trick only escapes <, >, & — not " or '.
    return (str || '')
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#39;');
  }

  // SEC: CSS-escape a string for safe use in querySelector attribute selectors.
  // Prevents CSS selector injection via model IDs containing " [ ] etc.
  function cssSafeAttr(str) {
    return (str || '').replace(/(["\\\[\](){}|^$*+?.#>~=!:,;])/g, '\\$1');
  }

  // SEC: Safe DOM ID from arbitrary string (for element IDs derived from model IDs etc.)
  function safeId(str) {
    return (str || '').replace(/[^a-zA-Z0-9]/g, '_');
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

  function timeAgo(ts) {
    var secs = Math.round((Date.now() - ts) / 1000);
    if (secs < 5) return I18n.t('time.just_now');
    if (secs < 60) return I18n.t('time.seconds_ago', { n: secs });
    var mins = Math.floor(secs / 60);
    if (mins < 60) return I18n.t('time.minutes_ago', { n: mins });
    var hrs = Math.floor(mins / 60);
    if (hrs < 24) return I18n.t('time.hours_minutes_ago', { h: hrs, m: mins % 60 });
    return I18n.t('time.days_ago', { n: Math.floor(hrs / 24) });
  }

  function capitalize(s) { return s.charAt(0).toUpperCase() + s.slice(1); }

  function setTierBadge(elementId, tier) {
    var el = document.getElementById(elementId);
    if (!el) return;
    el.textContent = capitalize(tier);
    el.className = 'tier-badge ' + tier.toLowerCase();
  }

  function renderSparkline(containerId, data) {
    var container = document.getElementById(containerId);
    if (!container || !data || data.length === 0) return;
    var hasActivity = data.some(function(v) { return v !== 0; });
    if (!hasActivity) { container.innerHTML = '<span class="text-muted" style="font-size:0.7rem">' + U.escapeHtml(I18n.t('chat.credit_activity_empty')) + '</span>'; return; }
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
    var match = S._modelDropdownData.find(function(m) { return m.id === modelId; });
    if (!match) return 'local';
    if (match.group === 'local') return 'local';
    if (match.group === 'swarm') return 'swarm';
    return 'cloud';
  }

  // Format a raw model ID into a friendly display name
  function formatModelDisplayName(id, opts) {
    if (!id) return I18n.t('utils.unknown_model');
    var name = id;
    name = name.replace(/\.gguf$/i, '').replace(/-gguf$/i, '');
    var parts = name.split(/[_]/);
    if (parts.length >= 2) {
      var prefix = parts[0].toLowerCase();
      var rest = parts.slice(1).join('_').toLowerCase();
      if (rest.indexOf(prefix) === 0) {
        name = parts.slice(1).join('_');
      }
    }
    name = name.replace(/(\d)\.(\d)/g, '$1\x00$2');
    var hideQuant = (opts && opts.hideQuant) || false;
    return name.split(/[-_.]/).filter(Boolean).map(function(s) {
      s = s.replace(/\x00/g, '.');
      if (/^(q\d|iq\d|f16|f32|bf16)/i.test(s)) return hideQuant ? null : s.toUpperCase();
      if (/^v\d/i.test(s)) return s;
      if (/^\d+\.?\d*[bBmM]$/.test(s)) return s.toUpperCase();
      if (hideQuant && /^[kms]$/i.test(s)) return null;
      return s.charAt(0).toUpperCase() + s.slice(1);
    }).filter(Boolean).join(' ');
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
        if (size === 1) msgs[k].classList.add('group-solo');
        else if (k === i) msgs[k].classList.add('group-first');
        else if (k === j) msgs[k].classList.add('group-last');
        else msgs[k].classList.add('group-mid');
      }
      i = j + 1;
    }
  }

  function appendMessageToDOM(role, content, isHtml, opts) {
    opts = opts || {};
    var container = document.getElementById('chat-messages');
    var empty = document.getElementById('chat-empty');
    if (empty) empty.style.display = 'none';

    var tmpl = document.getElementById('tmpl-chat-message');
    var div = tmpl.content.cloneNode(true).firstElementChild;
    div.classList.add(role);

    // Avatar
    var avatarEl = div.querySelector('.msg-avatar');
    if (role === 'assistant') {
      var _sess = S.currentSessionId && S.sessions[S.currentSessionId] ? S.sessions[S.currentSessionId] : null;
      var modelId = opts.model || (_sess ? _sess.model : '') || S.currentModel || '';
      var source = getModelSource(modelId);
      div.classList.add('source-' + source);
      var _avatarProvider = (modelId && S._modelDropdownData.find(function(m) { return m.id === modelId; }) || {}).group || null;
      var _iconKey = (_avatarProvider && _ICON_MAP[_avatarProvider]) ? _avatarProvider : modelIconKey(modelId);
      var _iconUrl = _iconKey ? providerIconUrl(_iconKey) : null;
      if (_iconUrl) {
        var img = document.createElement('img');
        img.src = _iconUrl; img.width = 16; img.height = 16; img.alt = '';
        img.className = 'provider-icon provider-avatar-icon'; img.style.display = 'block';
        avatarEl.appendChild(img);
      } else {
        avatarEl.textContent = I18n.t('chat.avatar_ai');
      }
    } else {
      var userImg = document.createElement('img');
      userImg.src = '/static/icons/swarm.svg'; userImg.alt = '';
      userImg.style.cssText = 'width:16px;height:16px;display:block';
      avatarEl.appendChild(userImg);
    }

    // Role label + source badge + encryption lock
    var roleEl = div.querySelector('.msg-role');
    roleEl.textContent = role === 'user' ? I18n.t('chat.role_user') : I18n.t('chat.role_assistant');
    if (role === 'assistant') {
      var sourceLabel = source === 'local' ? I18n.t('chat.source_local') : source === 'cloud' ? I18n.t('chat.source_cloud') : I18n.t('chat.source_network');
      var sourceBadge = document.createElement('span');
      sourceBadge.className = 'msg-source-badge source-' + source;
      sourceBadge.textContent = sourceLabel;
      roleEl.appendChild(sourceBadge);
    }
    if (opts && opts.encrypted) {
      var lockSpan = document.createElement('span');
      lockSpan.className = 'msg-enc-lock';
      lockSpan.title = I18n.t('chat.encrypted_title');
      lockSpan.innerHTML = '&#128274;';
      roleEl.appendChild(lockSpan);
    }

    // Content
    var contentEl = div.querySelector('.msg-content');
    if (isHtml) { contentEl.innerHTML = content; }
    else { contentEl.textContent = content; }

    // Assistant action buttons
    if (role === 'assistant') {
      var actions = document.createElement('div');
      actions.className = 'msg-actions';
      var copyBtn = document.createElement('button');
      copyBtn.className = 'msg-action-btn'; copyBtn.dataset.action = 'copy';
      copyBtn.title = I18n.t('chat.copy_response'); copyBtn.textContent = I18n.t('actions.copy');
      var compareBtn = document.createElement('button');
      compareBtn.className = 'msg-action-btn'; compareBtn.dataset.action = 'compare';
      compareBtn.title = I18n.t('chat.compare_question'); compareBtn.textContent = I18n.t('actions.try_other_models');
      actions.appendChild(copyBtn);
      actions.appendChild(compareBtn);
      div.querySelector('.msg-bubble').appendChild(actions);
    }

    container.appendChild(div);
    applyMessageGrouping(container);
    App.chat.scrollToBottom();
    return div;
  }

  function createEmptyState() {
    var div = document.createElement('div');
    div.className = 'chat-empty';
    div.id = 'chat-empty';

    var modelName = '';
    var modelData = null;
    var item = null;
    if (S.currentModel) {
      item = S._modelDropdownData.find(function(m) { return m.id === S.currentModel; });
      modelName = item ? item.name : S.currentModel;
      modelData = (App.data.cache.models || []).find(function(m) { return m.id === S.currentModel; });
    }

    var title = modelName ? I18n.t('chat.title_with_model', { model: escapeHtml(modelName) }) : I18n.t('chat.empty_title');
    var _emIconKey = S.currentModel ? ((item && item.group && _ICON_MAP[item.group]) ? item.group : modelIconKey(S.currentModel)) : null;
    var _emIconUrl = _emIconKey ? providerIconUrl(_emIconKey) : null;
    var icon = _emIconUrl
      ? '<img src="' + _emIconUrl + '" width="48" height="48" alt="" style="opacity:0.55;border-radius:10px;">'
      : '&#11088;';

    var encHint = '';
    if (modelData && modelData.encrypted_pipeline && modelData.shard_count > 1) {
      var isFullLocal = modelData.hosted_shards === modelData.shard_count;
      if (isFullLocal) {
        encHint = '<div class="chat-empty-hint" style="margin:6px 0;font-size:0.8rem;color:var(--green)">' +
          '&#128274; ' + escapeHtml(I18n.t('chat.enc_local_hint')) + '</div>';
      } else {
        encHint = '<div class="chat-empty-hint" style="margin:6px 0;font-size:0.8rem;color:var(--cyan,#22d3ee)">' +
          '&#128274; ' + escapeHtml(I18n.t('chat.enc_e2e_hint')) +
          '<br><span style="font-size:0.75rem;color:var(--text-muted)">' + escapeHtml(I18n.t('chat.enc_e2e_latency')) + '</span></div>';
      }
    } else if (modelData && modelData.shard_count > 1 && modelData.hosted_shards < modelData.shard_count) {
      encHint = '<div class="chat-empty-hint" style="margin:6px 0;font-size:0.8rem;color:var(--text-muted)">' +
        '&#127760; ' + escapeHtml(I18n.t('chat.enc_distributed_hint')) +
        '<br><span style="font-size:0.75rem">' + escapeHtml(I18n.t('chat.enc_enable_hint')) + '</span></div>';
    }

    div.innerHTML = '<div class="chat-empty-icon">' + icon + '</div>' +
      '<div class="chat-empty-title">' + title + '</div>' +
      encHint +
      '<div class="chat-empty-hint" style="margin:8px 0">' + I18n.t('chat.type_to_send') + '</div>' +
      '<div class="chat-empty-hint" style="font-size:0.8rem;margin-top:4px">' +
        (modelName ? '' : escapeHtml(I18n.t('chat.pick_model_hint')) + ' \u2022 ') +
        '<kbd>' + escapeHtml(I18n.t('chat.shift_enter')) + '</kbd></div>';
    return div;
  }

  function autoResizeInput() {
    if (!S.inputEl) S.inputEl = document.getElementById('chat-input');
    if (!S.inputEl) return;
    S.inputEl.style.height = 'auto';
    S.inputEl.style.height = Math.min(S.inputEl.scrollHeight, 200) + 'px';
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
    el.textContent = I18n.t('chat.word_count', { count: words });
    if (tokens > 7000) { el.className = 'token-counter danger'; el.title = I18n.t('chat.length_very_long'); }
    else if (tokens > 3000) { el.className = 'token-counter warn'; el.title = I18n.t('chat.length_long'); }
    else { el.className = 'token-counter'; el.title = I18n.t('chat.length_normal'); }
  }

  function updateChatAvailability(hasModels) {
    var sendBtn = document.getElementById('send-btn');
    var chatInput = document.getElementById('chat-input');
    var emptyState = document.querySelector('#chat-messages .chat-empty');

    if (sendBtn) sendBtn.disabled = !hasModels;
    if (chatInput) {
      chatInput.disabled = !hasModels;
      if (hasModels) {
        chatInput.placeholder = I18n.t('chat.placeholder') || 'Type your message...';
      } else {
        var dlInfo = document.getElementById('chat-dl-progress');
        if (!dlInfo) chatInput.placeholder = I18n.t('chat.no_models_placeholder') || 'Waiting for models — connect to peers or add a cloud provider in Settings';
      }
    }
    if (emptyState && !hasModels) {
      // Check if we have peers — different message for connected vs isolated
      var peerCount = (App.data.cache && App.data.cache.stats) ? (App.data.cache.stats.peers || 0) : 0;
      if (peerCount > 0) {
        // Connected to peers but no models ready yet — they're coming
        emptyState.innerHTML = '<div class="chat-empty-icon" style="font-size:2rem">' +
          '<div class="spinner" style="width:24px;height:24px;display:inline-block"></div></div>' +
          '<div class="chat-empty-title" style="font-size:1.1rem">' + (I18n.t('chat.discovering') || 'Discovering models on the network...') + '</div>' +
          '<div class="chat-empty-hint" style="margin:8px 0">' + (I18n.t('chat.discovering_hint') || 'Connected to ' + peerCount + ' peer' + (peerCount !== 1 ? 's' : '') + '. Models will appear in the dropdown as they become available.') + '</div>';
      } else {
        // No peers, no models — need to connect or add cloud provider
        emptyState.innerHTML = '<div class="chat-empty-icon">&#11203;</div>' +
          '<div class="chat-empty-title" style="font-size:1.1rem">' + (I18n.t('chat.getting_started') || 'Getting Started') + '</div>' +
          '<div class="chat-empty-hint" style="margin:8px 0">' + (I18n.t('chat.getting_started_hint') || 'Connect to the SwarmLLM network to access models, or add a cloud provider in Settings for instant chat.') + '</div>' +
          '<div style="display:flex;gap:8px;margin-top:12px;justify-content:center">' +
            '<button class="btn btn-primary" data-goto-network-code="1">' + (I18n.t('chat.connect_peers') || 'Connect to Peers') + '</button>' +
            '<button class="btn btn-outline" data-goto-settings="1" style="border:1px solid var(--border)">' + (I18n.t('chat.add_provider') || 'Add Cloud Provider') + '</button>' +
          '</div>';
      }
    }
  }

  function updateChatDownloadProgress(acquisitions) {
    var container = document.querySelector('.chat-input-area');
    if (!container) return;
    var existing = document.getElementById('chat-dl-progress');

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
    var text = I18n.t('chat.downloading_progress', { name: name, pct: pct });
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

  // --- Unified API error handler ---
  // Extracts error message from response body and shows a toast.
  // Usage: var ok = await App.utils.handleApiError(resp, 'Action failed');
  // Returns true if response was ok, false if error was shown.
  async function handleApiError(resp, fallbackMsg) {
    if (resp && resp.ok) return true;
    var msg = fallbackMsg || I18n.t('common.request_failed');
    try {
      var body = await resp.json();
      if (body && body.error) {
        msg = body.error.message || body.error || msg;
      }
    } catch (e) {}
    if (App.notifications && App.notifications.showToast) {
      App.notifications.showToast(msg, 'error', 6000);
    }
    return false;
  }

  // --- Extract error message string from a failed response ---
  // Parses the JSON body and returns the error message string, or fallback.
  // Usage: var msg = await App.utils.getApiErrorMessage(resp, 'Action failed');
  async function getApiErrorMessage(resp, fallback) {
    var msg = fallback || I18n.t('common.request_failed');
    try {
      var body = await resp.json();
      if (body && body.error) {
        msg = body.error.message || body.error || msg;
      }
    } catch (e) {}
    return msg;
  }

  // Export utilities
  App.utils = {
    escapeHtml: escapeHtml,
    cssSafeAttr: cssSafeAttr,
    safeId: safeId,
    formatUptime: formatUptime,
    formatMB: formatMB,
    formatBytes: formatBytes,
    formatSpeed: formatSpeed,
    formatEta: formatEta,
    timeAgo: timeAgo,
    capitalize: capitalize,
    setTierBadge: setTierBadge,
    renderSparkline: renderSparkline,
    getModelSource: getModelSource,
    formatModelDisplayName: formatModelDisplayName,
    applyMessageGrouping: applyMessageGrouping,
    appendMessageToDOM: appendMessageToDOM,
    createEmptyState: createEmptyState,
    autoResizeInput: autoResizeInput,
    updateTokenCounter: updateTokenCounter,
    updateChatAvailability: updateChatAvailability,
    updateChatDownloadProgress: updateChatDownloadProgress,
    handleApiError: handleApiError,
    getApiErrorMessage: getApiErrorMessage,
  };
})();
