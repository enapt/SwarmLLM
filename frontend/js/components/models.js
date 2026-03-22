'use strict';

// ============================================================================
// SwarmLLM — Models Component
// Model loading, dropdown, HF search/download, shard menu, metadata,
// auto-manage, cancel/remove/unload, shutdown
// ============================================================================

(function() {
  var S = App.state;
  var U = App.utils;

  // ========================================================================
  // HuggingFace Module
  // ========================================================================
  App.hf = {
    search: async function() {
      var query = document.getElementById('hf-search-input').value.trim();
      var suggestions = document.getElementById('hf-suggestions');
      if (!query) {
        if (suggestions) suggestions.style.display = '';
        return;
      }

      var results = document.getElementById('hf-results');
      var loading = document.getElementById('hf-loading');
      results.innerHTML = '';
      if (suggestions) suggestions.style.display = 'none';
      loading.classList.remove('hidden');

      try {
        var resp = await App.authFetch('/api/admin/hf/search?query=' + encodeURIComponent(query));
        loading.classList.add('hidden');

        if (!resp.ok) {
          var errBody = await resp.text();
          try { var errJson = JSON.parse(errBody); errBody = errJson.error ? errJson.error.message : errBody; } catch (e2) {}
          results.innerHTML = '<div class="empty-state"><p>Search failed: ' + U.escapeHtml(errBody) + '</p></div>';
          return;
        }

        var data = await resp.json();

        if (!Array.isArray(data) || data.length === 0) {
          results.innerHTML = '<div class="empty-state"><p>No GGUF models found for "' + U.escapeHtml(query) + '"</p></div>';
          return;
        }

        // Store data for re-sorting
        App.hf._lastData = data;
        App.hf._renderResults(data);
      } catch (e) {
        loading.classList.add('hidden');
        results.innerHTML = '<div class="empty-state"><p>Search failed: ' + U.escapeHtml(e.message) + '</p></div>';
      }
    },

    _lastData: null,

    sortResults: function() {
      if (!App.hf._lastData) return;
      var sortBy = (document.getElementById('hf-sort') || {}).value || 'score';
      var data = App.hf._lastData.slice();
      if (sortBy === 'downloads') data.sort(function(a,b) { return (b.downloads||0) - (a.downloads||0); });
      else if (sortBy === 'size_asc') data.sort(function(a,b) { return (a.est_shard_size||0) - (b.est_shard_size||0); });
      else if (sortBy === 'size_desc') data.sort(function(a,b) { return (b.est_shard_size||0) - (a.est_shard_size||0); });
      else data.sort(function(a,b) { return (b.composite_score||0) - (a.composite_score||0); });
      App.hf._renderResults(data);
    },

    _renderResults: function(data) {
        var results = document.getElementById('hf-results');
        results.innerHTML = '';
        var hfTmpl = document.getElementById('tmpl-hf-result-card');
        data.forEach(function(repo) {
          var card = hfTmpl.content.cloneNode(true).firstElementChild;
          var safeKey = (repo.repo_id || '').replace(/[^a-zA-Z0-9]/g, '_');
          var variants = repo.variants || [];
          var recommended = repo.recommended_variant || '';

          // Name
          card.querySelector('.hf-model-name').textContent = repo.repo_id;

          // Stats meta (downloads, likes, VRAM fit)
          var statsHtml = '';
          if (repo.downloads) statsHtml += '<span>' + repo.downloads.toLocaleString() + ' downloads</span>';
          if (repo.likes) statsHtml += '<span>' + repo.likes.toLocaleString() + ' likes</span>';
          var shardSizeStr = repo.est_shard_size ? U.formatBytes(repo.est_shard_size) : '';
          var boomerangSizeStr = repo.est_boomerang_size ? U.formatBytes(repo.est_boomerang_size) : '';
          if (repo.fits_boomerang) {
            statsHtml += '<span><span style="color:var(--green)" title="First+last shard fit VRAM (~' + boomerangSizeStr + ')">&#9989; Run locally</span></span>';
          } else if (repo.fits_shard) {
            statsHtml += '<span><span style="color:var(--cyan)" title="Individual shards fit VRAM (~' + shardSizeStr + '/shard)">&#128279; Can host shards</span></span>';
          } else if (repo.fits_vram === false && variants.length > 0) {
            statsHtml += '<span><span style="color:var(--orange)" title="Even individual shards may exceed your available VRAM">&#9888; Exceeds VRAM</span></span>';
          }
          // Composite score badge
          if (repo.composite_score != null) {
            var scoreColor = repo.composite_score >= 60 ? 'var(--green)' : repo.composite_score >= 30 ? 'var(--yellow)' : 'var(--text-muted)';
            statsHtml += '<span style="color:' + scoreColor + '; font-weight:600" title="Fit score: quality=' + ((repo.score_breakdown||{}).quality||0) + '% fit=' + ((repo.score_breakdown||{}).fit||0) + '% demand=' + ((repo.score_breakdown||{}).demand||0) + '% size=' + ((repo.score_breakdown||{}).size||0) + '%">' + repo.composite_score + ' pts</span>';
          }
          card.querySelector('.hf-meta-stats').innerHTML = statsHtml;

          // Network meta
          var replicas = repo.network_replicas || 0;
          var networkHtml = replicas > 0
            ? '<span class="badge-swarm" title="' + replicas + ' node(s) already hosting this model on the swarm">On Swarm &mdash; ' + replicas + ' node' + (replicas !== 1 ? 's' : '') + '</span>'
            : '<span class="badge-new" title="Not yet on the swarm">New to network</span>';
          if (replicas === 0) networkHtml += '<span style="color:var(--green)" title="No replicas yet — high credit earning potential">&#128176; High demand</span>';
          else if (replicas <= 2) networkHtml += '<span style="color:var(--yellow)" title="Few replicas — good credit earning potential">&#128176; Medium demand</span>';
          else networkHtml += '<span style="color:var(--text-muted)" title="Well replicated across the network">&#128176; Well replicated</span>';
          card.querySelector('.hf-meta-network').innerHTML = networkHtml;

          // Variant selector
          var selectEl = card.querySelector('.hf-quant-select');
          if (variants.length > 1) {
            selectEl.removeAttribute('hidden');
            selectEl.id = 'quant-' + safeKey;
            variants.forEach(function(v) {
              var opt = document.createElement('option');
              opt.value = v.filename;
              var label = v.quant + (v.size_bytes ? ' \u2014 ' + U.formatBytes(v.size_bytes) : '');
              if (v.quant === recommended) { label += ' (Recommended)'; opt.selected = true; }
              opt.textContent = label;
              selectEl.appendChild(opt);
            });
          } else {
            selectEl.remove();
          }

          // Download button
          var dlBtn = card.querySelector('.hf-dl-btn');
          dlBtn.setAttribute('data-hf-download', repo.repo_id);
          dlBtn.setAttribute('data-hf-variant', safeKey);
          if (variants.length === 1) dlBtn.setAttribute('data-hf-filename', variants[0].filename);

          results.appendChild(card);
        });
    },

    download: async function(repoId, variantKey) {
      try {
        var filename = '';
        if (variantKey) {
          var quantEl = document.getElementById('quant-' + variantKey);
          if (quantEl) {
            filename = quantEl.value;
          }
        }
        if (!filename) {
          var btn = document.querySelector('[data-hf-download="' + repoId + '"]');
          filename = btn ? (btn.getAttribute('data-hf-filename') || '') : '';
        }
        if (!filename) {
          App.ui.showBanner('error', 'No model variant selected');
          return;
        }

        App.ui.showBanner('info', 'Checking model availability...');
        var resp = await App.authFetch('/api/admin/hf/download-shards', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ repo_id: repoId, filename: filename, peer_fair_share: true }),
        });
        var data = await resp.json();
        if (!resp.ok) {
          var errMsg = (data.error && data.error.message) || 'Download failed';
          App.ui.showBanner('error', errMsg);
          return;
        }
        if (data.status === 'started') {
          App.notifications.showToast('Download started \u2014 model data will be ready soon', 'success');
          App.ui.closeModelBrowser();
        } else {
          App.notifications.showToast(data.message || 'Download could not be started', 'warning');
        }
      } catch (e) {
        App.ui.showBanner('error', 'Download failed: ' + e.message);
      }
    }
  };

  // ========================================================================
  // Model loading + selection + dropdown
  // ========================================================================
  App.models = {
    load: async function() {
      try {
        if (App.settings._apiKeyPromise) await App.settings._apiKeyPromise;
        var result = await App.data.loadModels();
        var adminModels = result.models;
        var providerModels = result.cloudModels;

        App.dashboard.renderModels(adminModels, providerModels);

        var readySet = {};
        adminModels.forEach(function(m) {
          var isReady = m.status === 'loaded' || m.status === 'ready' ||
            (m.global_available === m.shard_count && m.shard_count > 0);
          if (isReady) readySet[m.id] = true;
        });

        var readyModels = adminModels.filter(function(m) { return readySet[m.id]; });
        var hasAny = readyModels.length > 0 || providerModels.length > 0;

        var groups = [];
        S._modelDropdownData = [];

        if (readyModels.length > 0) {
          var localItems = [];
          var swarmItems = [];
          readyModels.forEach(function(m) {
            var displayName = U.formatModelDisplayName(m.name || m.id);
            var isDistributed = m.shard_count > 0 && (m.hosted_shards || 0) < m.shard_count;
            var item = { id: m.id, name: displayName.length > 40 ? displayName.substring(0, 40) + '...' : displayName, group: isDistributed ? 'swarm' : 'local', encrypted: !!m.encrypted_pipeline };
            if (isDistributed) { swarmItems.push(item); } else { localItems.push(item); }
          });
          if (localItems.length > 0) {
            groups.push({ key: 'local', label: 'On this computer', items: localItems });
            S._modelDropdownData = S._modelDropdownData.concat(localItems);
          }
          if (swarmItems.length > 0) {
            groups.push({ key: 'swarm', label: 'Swarm network', items: swarmItems });
            S._modelDropdownData = S._modelDropdownData.concat(swarmItems);
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
            items.sort(function(a, b) {
              var na = a.name.toLowerCase(), nb = b.name.toLowerCase();
              return na < nb ? -1 : na > nb ? 1 : 0;
            });
            groups.push({ key: p, label: (PROVIDER_NAMES[p] || p) + ' (cloud)', items: items });
            S._modelDropdownData = S._modelDropdownData.concat(items);
          });
        }

        App.models.renderDropdown(groups, hasAny);

        if (hasAny) {
          var allIds = S._modelDropdownData.map(function(m) { return m.id; });
          var sessionModel = S.currentSessionId && S.sessions[S.currentSessionId] ? S.sessions[S.currentSessionId].model : null;
          var savedModel = null;
          try { savedModel = localStorage.getItem('swarmllm_current_model'); } catch (e) {}
          var preferred = sessionModel || savedModel;
          var found = preferred && allIds.indexOf(preferred) !== -1;
          App.models.selectDropdown(found ? preferred : allIds[0], { silent: true });
        } else {
          S.currentModel = '';
          updateModelDropdownLabel('Select model...');
        }

        App.models.syncMobile();
        U.updateChatAvailability(hasAny);
        if (App.chat && App.chat.updateChatHeader) App.chat.updateChatHeader();
        if (App.chat && App.chat.renderSessionList) App.chat.renderSessionList();
        if (App.chat && App.chat.renderMessages && S.currentSessionId && S.sessions[S.currentSessionId] && S.sessions[S.currentSessionId].messages.length === 0) {
          App.chat.renderMessages();
        }
      } catch (e) {
        App.ui.showBanner('error', I18n.t('errors.server_unreachable'));
      }
    },

    renderDropdown: function(groups, hasAny) {
      var list = document.getElementById('model-dropdown-list');
      if (!list) return;
      list.innerHTML = '';

      if (!hasAny) {
        var peerCount = (App.data.cache && App.data.cache.stats) ? (App.data.cache.stats.peer_count || 0) : 0;
        if (peerCount > 0) {
          list.innerHTML = '<div class="model-dropdown-empty">' +
            (I18n.t('models.discovering') || 'Discovering models...') +
            '<br><span style="font-size:0.72rem;color:var(--text-muted)">' +
            (I18n.t('models.discovering_hint') || 'Connected to ' + peerCount + ' peers. Models will appear as the network syncs.') +
            '</span></div>';
        } else {
          list.innerHTML = '<div class="model-dropdown-empty">' +
            (I18n.t('models.no_models') || 'No models available yet') +
            '<br><span style="font-size:0.72rem;color:var(--text-muted)">' +
            (I18n.t('models.no_models_hint') || 'Connect to the network to access shared models, or add a cloud provider in Settings') +
            '</span></div>';
        }
        return;
      }

      groups.forEach(function(g) {
        var groupEl = document.createElement('div');
        groupEl.className = 'model-dropdown-group';
        groupEl.setAttribute('data-group', g.key);

        var header = document.createElement('div');
        header.className = 'model-dropdown-group-header';
        var groupIconHtml = providerIconHtml(g.key, 14);
        header.innerHTML = '<span class="group-arrow">&#9662;</span>' + (groupIconHtml ? ' ' + groupIconHtml : '') + ' ' + U.escapeHtml(g.label) + ' <span style="opacity:0.5;font-weight:400">(' + g.items.length + ')</span>';
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
          var nameSpan = document.createElement('span');
          nameSpan.textContent = item.name;
          if (item.id !== item.name) el.setAttribute('title', item.id);
          el.appendChild(nameSpan);
          if (item.meta) {
            var metaParts = [];
            var m = item.meta;
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
              metaSpan.textContent = metaParts.join(' \u00b7 ');
              el.appendChild(metaSpan);
            }
            el.title = item.id + '\n' + JSON.stringify(item.meta, null, 2);
          } else {
            el.title = item.id;
          }
          el.addEventListener('click', function() {
            var prevM = S.currentModel;
            App.models.selectDropdown(item.id);
            App.models.closeDropdown();
            if (item.id !== prevM || !S.currentSessionId || !S.sessions[S.currentSessionId]) {
              App.chat.newSession();
            }
            App.ui.switchTab('chat');
          });
          itemsEl.appendChild(el);
        });
        groupEl.appendChild(itemsEl);
        list.appendChild(groupEl);
      });
    },

    selectDropdown: function(modelId, opts) {
      opts = opts || {};
      var prevModel = S.currentModel;
      S.currentModel = modelId;
      document.getElementById('model-select').value = modelId;
      try { localStorage.setItem('swarmllm_current_model', modelId); } catch (e) {}

      var item = S._modelDropdownData.find(function(m) { return m.id === modelId; });
      updateModelDropdownLabel(item ? item.name : modelId);

      var items = document.querySelectorAll('#model-dropdown-list .model-dropdown-item');
      items.forEach(function(el) {
        el.classList.toggle('selected', el.getAttribute('data-value') === modelId);
      });

      var trigger = document.getElementById('model-dropdown-trigger');
      if (trigger) {
        trigger.classList.remove('flash');
        void trigger.offsetWidth;
        trigger.classList.add('flash');
      }

      if (!opts.silent && prevModel && prevModel !== modelId && S.currentSessionId && S.sessions[S.currentSessionId]) {
        var s = S.sessions[S.currentSessionId];
        if (s.messages.length > 0) {
          App.chat.newSession();
          App.notifications.showToast('New session started for ' + U.formatModelDisplayName(modelId), 'info');
        } else {
          s.model = modelId;
          App.chat.saveSessions();
          App.chat.renderMessages();
          App.chat.updateChatHeader();
          App.chat.renderSessionList();
        }
      }
    },

    closeDropdown: function() {
      var dd = document.getElementById('model-dropdown');
      if (dd) dd.classList.remove('open');
    },

    initDropdown: function() {
      var trigger = document.getElementById('model-dropdown-trigger');
      var dd = document.getElementById('model-dropdown');
      var search = document.getElementById('model-dropdown-search');
      if (!trigger || !dd) return;

      trigger.addEventListener('click', function(e) {
        e.stopPropagation();
        dd.classList.toggle('open');
        if (dd.classList.contains('open') && search) {
          search.value = '';
          App.models.filterDropdown('');
          setTimeout(function() { search.focus(); }, 50);
        }
      });

      if (search) {
        search.addEventListener('input', function() {
          App.models.filterDropdown(search.value);
        });
        search.addEventListener('keydown', function(e) {
          if (e.key === 'Escape') { App.models.closeDropdown(); }
          if (e.key === 'Enter') {
            var first = document.querySelector('#model-dropdown-list .model-dropdown-item:not(.hidden)');
            if (first) {
              App.models.selectDropdown(first.getAttribute('data-value'));
              App.models.closeDropdown();
            }
          }
        });
      }

      document.addEventListener('click', function(e) {
        if (!dd.contains(e.target)) App.models.closeDropdown();
      });
    },

    filterDropdown: function(query) {
      var q = query.toLowerCase().trim();
      var items = document.querySelectorAll('#model-dropdown-list .model-dropdown-item');
      items.forEach(function(el) {
        var match = !q || el.getAttribute('data-search').indexOf(q) !== -1;
        el.classList.toggle('hidden', !match);
      });
      var groups = document.querySelectorAll('#model-dropdown-list .model-dropdown-group');
      groups.forEach(function(g) {
        var visibleItems = g.querySelectorAll('.model-dropdown-item:not(.hidden)');
        if (q) {
          g.classList.toggle('collapsed', visibleItems.length === 0);
        }
      });
    },

    request: async function(modelId) {
      try {
        var resp = await App.authFetch('/api/admin/models/' + encodeURIComponent(modelId) + '/add', { method: 'POST' });
        var data = await resp.json();
        if (data.status === 'acquiring') {
          S.activeAcquisitions[modelId] = { started: Date.now() };
          App.dashboard.renderAcquisitionPanel(modelId, null);
        } else {
          App.ui.showBanner('warning', data.message || 'Model download unavailable');
        }
      } catch (e) {
        App.ui.showBanner('error', 'Failed to request model: ' + e.message);
      }
    },

    select: function(modelId) {
      App.models.selectDropdown(modelId);
      App.ui.showBanner('success', 'Model selected: ' + modelId);
      App.models.load();
    },

    cancelDownload: async function(modelId) {
      if (!confirm(I18n.t('actions.confirm_cancel_download', { model: modelId }))) return;
      try {
        var resp = await App.authFetch('/api/admin/downloads/' + encodeURIComponent(modelId) + '/cancel', { method: 'POST' });
        if (resp.ok) {
          App.ui.showBanner('success', 'Download cancelled');
          var card = document.querySelector('[data-model-id="' + U.cssSafeAttr(modelId) + '"]');
          if (card) {
            var progress = card.querySelector('.dl-progress');
            if (progress) progress.remove();
            card.classList.remove('downloading');
            card.querySelectorAll('.shard-cell.downloading, .shard-cell.verifying').forEach(function(cell) {
              var idx = cell.getAttribute('data-shard-index') || cell.textContent;
              var cPreserve = '';
              if (cell.classList.contains('locked')) cPreserve += ' locked';
              if (cell.classList.contains('shard-endpoint')) cPreserve += ' shard-endpoint';
              if (cell.classList.contains('shard-pinned')) cPreserve += ' shard-pinned';
              cell.className = 'shard-cell missing' + cPreserve;
              Array.from(cell.childNodes).forEach(function(n) { if (n.nodeType === 3) n.textContent = ''; });
              cell.insertBefore(document.createTextNode(idx), cell.firstChild);
              cell.style.removeProperty('--dl-pct');
            });
          }
          delete S.activeAcquisitions[modelId];
          setTimeout(function() { App.dashboard.loadInitial(); }, 1000);
        } else {
          var errData = await resp.json().catch(function() { return {}; });
          App.ui.showBanner('error', errData.error ? errData.error.message : 'Failed to cancel download');
        }
      } catch (e) {
        App.ui.showBanner('error', 'Cancel failed: ' + e.message);
      }
    },

    remove: async function(modelId) {
      if (!confirm(I18n.t('actions.confirm_remove_model', { model: modelId }))) return;
      try {
        var resp = await App.authFetch('/api/admin/models/' + encodeURIComponent(modelId), { method: 'DELETE' });
        if (resp.ok) {
          App.ui.showBanner('success', 'Model removed: ' + modelId);
          var card = document.querySelector('[data-model-id="' + U.cssSafeAttr(modelId) + '"]');
          if (card) card.remove();
          setTimeout(function() { App.dashboard.loadInitial(); }, 1000);
        } else {
          var errData = await resp.json().catch(function() { return {}; });
          App.ui.showBanner('error', errData.error ? errData.error.message : 'Failed to remove model');
        }
      } catch (e) {
        App.ui.showBanner('error', 'Remove failed: ' + e.message);
      }
    },

    unload: async function(modelId) {
      try {
        var resp = await App.authFetch('/api/admin/models/' + encodeURIComponent(modelId) + '/unload', { method: 'POST' });
        if (resp.ok) {
          var result = await resp.json().catch(function() { return {}; });
          var freedMb = result.estimated_freed_mb || 0;
          var name = result.model_name || U.formatModelDisplayName(modelId);
          var msg = name + ' unloaded from memory';
          if (freedMb > 0) msg += ' (~' + U.formatMB(freedMb) + ' freed)';
          App.notifications.showToast(msg, 'success');
          App.notifications.logActivity('\u{1F4A4}', msg);
          App.dashboard._logModelEvent(modelId, '\u{1F4A4}', 'Unloaded from memory' + (freedMb > 0 ? ' — ~' + U.formatMB(freedMb) + ' freed' : ''));
          App.models.load();
        } else {
          var errData = await resp.json().catch(function() { return {}; });
          App.notifications.showToast(errData.error ? errData.error.message : 'Failed to unload model', 'error');
        }
      } catch (e) {
        App.notifications.showToast('Unload failed: ' + e.message, 'error');
      }
    },

    toggleAutoManage: async function(modelId) {
      var card = document.querySelector('[data-model-id="' + U.cssSafeAttr(modelId) + '"]');
      if (!card) return;

      var existing = card.querySelector('.auto-manage-panel');
      if (existing) { existing.remove(); return; }

      var policy = { enabled: true, max_shards: 0, prune_enabled: true };
      var encStatus = { encrypted_pipeline: false, ready: false, has_first_shard: false, has_last_shard: false, shard_count: 0 };
      try {
        var results = await Promise.all([
          App.authFetch('/api/admin/models/' + encodeURIComponent(modelId) + '/auto-manage'),
          App.authFetch('/api/admin/models/' + encodeURIComponent(modelId) + '/encrypted-pipeline'),
        ]);
        if (results[0].ok) policy = await results[0].json();
        if (results[1].ok) encStatus = await results[1].json();
      } catch (e) {
        App.ui.showBanner('error', 'Could not load model policy');
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
          '<label><input type="checkbox" id="am-enabled-' + U.safeId(modelId) + '"' + (policy.enabled ? ' checked' : '') + '> Auto-manage enabled</label>' +
        '</div>' +
        '<div class="am-row">' +
          '<label><input type="checkbox" id="am-prune-' + U.safeId(modelId) + '"' + (policy.prune_enabled !== false ? ' checked' : '') + '> Auto-prune enabled</label>' +
        '</div>' +
        '<div class="am-row">' +
          '<label>Max shards:</label>' +
          '<input type="number" id="am-max-' + U.safeId(modelId) + '" value="' + (policy.max_shards || 0) + '" min="0" step="1">' +
          '<span class="text-muted" style="font-size:0.7rem">0 = unlimited</span>' +
        '</div>' +
        '<hr style="margin:0.3rem 0;border-color:var(--border)">' +
        '<div class="am-row" style="flex-direction:column;gap:0.2rem">' +
          '<label><input type="checkbox" id="am-encrypted-' + U.safeId(modelId) + '"' +
            (encStatus.encrypted_pipeline ? ' checked' : '') + encDisabled +
            '> &#128274; Encrypted pipeline</label>' +
          '<span class="' + encReadyClass + '" style="font-size:0.65rem">' + encReadyText + '</span>' +
          encOverheadNote +
        '</div>' +
        '<div class="am-row">' +
          '<button class="btn btn-sm btn-primary" data-am-save="' + U.escapeHtml(modelId) + '">Save</button>' +
        '</div>';
      card.appendChild(panel);
    },

    saveAutoManage: async function(modelId) {
      var sid = U.safeId(modelId);
      var enabledEl = document.getElementById('am-enabled-' + sid);
      var maxEl = document.getElementById('am-max-' + sid);
      var pruneEl = document.getElementById('am-prune-' + sid);
      var encryptedEl = document.getElementById('am-encrypted-' + sid);
      if (!enabledEl || !maxEl) return;

      try {
        var amResp = await App.authFetch('/api/admin/models/' + encodeURIComponent(modelId) + '/auto-manage', {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            enabled: enabledEl.checked,
            max_shards: parseInt(maxEl.value, 10) || 0,
            prune_enabled: pruneEl ? pruneEl.checked : true,
          }),
        });

        var encErr = null;
        if (encryptedEl && !encryptedEl.disabled) {
          var encResp = await App.authFetch('/api/admin/models/' + encodeURIComponent(modelId) + '/encrypted-pipeline', {
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
          App.ui.showBanner('success', 'Model policy saved');
          var card = document.querySelector('[data-model-id="' + U.cssSafeAttr(modelId) + '"]');
          var panel = card ? card.querySelector('.auto-manage-panel') : null;
          if (panel) panel.remove();
        } else {
          var errMsg = encErr || '';
          if (!amResp.ok) {
            var errData = await amResp.json().catch(function() { return {}; });
            errMsg = errData.error ? errData.error.message : 'Save failed';
          }
          App.ui.showBanner('error', errMsg);
        }
      } catch (e) {
        App.ui.showBanner('error', 'Save failed: ' + e.message);
      }
    },

    toggleMetadata: async function(modelId) {
      var panel = document.querySelector('[data-meta-panel="' + U.cssSafeAttr(modelId) + '"]');
      if (!panel) return;
      if (!panel.classList.contains('hidden')) { panel.classList.add('hidden'); return; }
      panel.classList.remove('hidden');
      if (panel.innerHTML) return;

      panel.innerHTML = '<div class="meta-loading"><span class="spinner" style="width:14px;height:14px;border-width:1.5px"></span> Loading metadata...</div>';
      try {
        var data = S.metadataCache[modelId];
        if (!data) {
          var resp = await App.authFetch('/api/admin/models/' + encodeURIComponent(modelId) + '/metadata');
          if (!resp.ok) throw new Error('Failed to load metadata');
          data = await resp.json();
          S.metadataCache[modelId] = data;
        }
        renderMetadataPanel(panel, data);
      } catch (e) {
        panel.innerHTML = '<div class="meta-error">Failed to load GGUF metadata</div>';
      }
    },

    shutdown: async function() {
      if (!confirm('Shut down SwarmLLM node?')) return;
      try {
        await App.authFetch('/api/admin/shutdown', { method: 'POST' });
        document.body.innerHTML = '<div style="display:flex;align-items:center;justify-content:center;height:100vh;color:var(--text-muted);font-size:1.2rem">SwarmLLM has been shut down.</div>';
      } catch (e) {
        App.ui.showBanner('error', 'Shutdown failed: ' + e.message);
      }
    },

    initMobileSync: function() {
      var mobile = document.getElementById('mobile-model-select');
      var mobileBtn = document.getElementById('btn-mobile-browse');

      if (mobile) {
        mobile.addEventListener('change', function() {
          App.models.selectDropdown(mobile.value);
        });
      }

      if (mobileBtn) {
        mobileBtn.addEventListener('click', function() {
          App.ui.openModelBrowser();
        });
      }
    },

    syncMobile: function() {
      var mobile = document.getElementById('mobile-model-select');
      if (!mobile) return;
      mobile.innerHTML = '';
      S._modelDropdownData.forEach(function(m) {
        var opt = document.createElement('option');
        opt.value = m.id;
        opt.textContent = m.name;
        mobile.appendChild(opt);
      });
      mobile.value = S.currentModel;
    }
  };

  function updateModelDropdownLabel(text) {
    var label = document.getElementById('model-dropdown-label');
    if (!label) return;
    var item = S._modelDropdownData.find(function(m) { return m.name === text || m.id === text; });
    label.textContent = text;
    var trigger = document.getElementById('model-dropdown-trigger');
    if (trigger && item) trigger.title = item.id;
    if (trigger) trigger.classList.toggle('no-model', !S.currentModel);
  }

  function renderMetadataPanel(panel, data) {
    var html = '<div class="meta-header">GGUF Metadata</div>';
    var g = data.general || {};
    var m = data.model || {};
    var summaryParts = [];
    if (g.architecture) {
      var archTag = '<span class="meta-tag">' + U.escapeHtml(g.architecture) + '</span>';
      if (g.architecture_supported === false) {
        archTag += '<span class="meta-tag" style="background:var(--error-bg,#5c2020);color:var(--error-fg,#ff6b6b)">unsupported</span>';
      }
      summaryParts.push(archTag);
    }
    if (g.quantization) summaryParts.push('<span class="meta-tag">' + U.escapeHtml(g.quantization) + '</span>');
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
        var val = typeof f[1] === 'number' ? f[1].toLocaleString() : U.escapeHtml(String(f[1]));
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
        if (f[1] != null) html += '<tr><td class="meta-key">' + U.escapeHtml(f[0]) + '</td><td class="meta-val">' + U.escapeHtml(String(f[1])) + '</td></tr>';
      });
      html += '</tbody></table>';
    }

    var tens = data.tensors || {};
    if (tens.count) html += '<div class="meta-tensor-info">' + tens.count + ' tensors, data offset: ' + U.formatBytes(tens.data_offset || 0) + '</div>';

    var raw = data.raw || [];
    if (raw.length > 0) {
      html += '<details class="meta-raw-details"><summary>All metadata keys (' + raw.length + ')</summary>';
      html += '<table class="meta-table meta-raw-table"><tbody>';
      raw.forEach(function(r) { html += '<tr><td class="meta-key">' + U.escapeHtml(r.key) + '</td><td class="meta-val">' + U.escapeHtml(r.value) + '</td></tr>'; });
      html += '</tbody></table></details>';
    }
    panel.innerHTML = html;
  }

  // ========================================================================
  // Shard Context Menu
  // ========================================================================
  App.shardMenu = {
    menu: null,
    currentModel: null,
    currentIndex: null,
    currentState: null,
    currentLocked: false,

    init: function() {
      this.menu = document.getElementById('shard-context-menu');
      // Wire up buttons once
      var loadBtn = document.getElementById('shard-ctx-load');
      if (loadBtn) loadBtn.addEventListener('click', function() { App.shardMenu.loadShard(); });
      var unloadBtn = document.getElementById('shard-ctx-unload');
      if (unloadBtn) unloadBtn.addEventListener('click', function() { App.shardMenu.unloadShard(); });
      var lockBtn = document.getElementById('shard-ctx-lock');
      if (lockBtn) lockBtn.addEventListener('click', function() { App.shardMenu.toggleLock(); });
    },

    show: function(modelId, shardIndex, shardState, x, y, isLocked, isInVram) {
      if (!this.menu) this.init();
      this.currentModel = modelId;
      this.currentIndex = shardIndex;
      this.currentState = shardState;
      this.currentLocked = !!isLocked;
      this.currentInVram = !!isInVram;

      var header = document.getElementById('shard-ctx-header');
      var statusEl = document.getElementById('shard-ctx-status');
      var btn = document.getElementById('shard-ctx-action');
      var unloadBtn = document.getElementById('shard-ctx-unload');
      var lockBtn = document.getElementById('shard-ctx-lock');
      var warnEl = document.getElementById('shard-ctx-warn');

      header.textContent = 'Part ' + (shardIndex + 1);

      // Status line
      var statusText = '';
      if (shardState === 'local' && isInVram) statusText = 'Active (loaded in memory)';
      else if (shardState === 'local') statusText = 'On disk (not loaded)';
      else if (shardState === 'downloading') statusText = 'Downloading...';
      else if (shardState === 'peer') statusText = 'Available from peers';
      else statusText = 'Not available';
      statusEl.textContent = statusText;

      // Primary action
      if (shardState === 'local') {
        btn.textContent = 'Delete from disk';
        btn.className = 'shard-ctx-btn danger';
      } else if (shardState === 'downloading') {
        btn.textContent = 'Cancel download';
        btn.className = 'shard-ctx-btn danger';
      } else {
        btn.textContent = 'Download this part';
        btn.className = 'shard-ctx-btn';
      }

      // Load button — only for local shards NOT in memory
      var loadBtn = document.getElementById('shard-ctx-load');
      if (loadBtn) {
        loadBtn.style.display = (shardState === 'local' && !isInVram) ? '' : 'none';
        loadBtn.title = 'Load this part into memory for inference. The model worker will restart to include it.';
      }

      // Unload button — only when shard is loaded in memory
      if (unloadBtn) {
        unloadBtn.style.display = (shardState === 'local' && isInVram) ? '' : 'none';
        unloadBtn.title = 'Keeps the file on disk but frees RAM/VRAM. The model worker will restart without this part.';
      }

      // Lock button — only for local shards
      if (lockBtn) {
        lockBtn.textContent = isLocked ? 'Unlock (unpin)' : 'Lock (pin)';
        lockBtn.style.display = (shardState === 'local') ? '' : 'none';
      }

      // Warning when auto-manage is on
      if (warnEl) {
        warnEl.style.display = 'none';
        if (shardState === 'local') {
          warnEl.innerHTML = '\u26a0 Auto-manage may re-download this part if demand is high';
          warnEl.style.display = '';
        }
      }

      var mw = 220, mh = 160;
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
        if (!confirm(I18n.t('actions.confirm_remove_shard', { index: idx, model: modelId }))) return;
        try {
          var resp = await App.authFetch('/api/admin/models/' + encodeURIComponent(modelId) + '/shards/' + idx, { method: 'DELETE' });
          if (resp.ok) {
            App.ui.showBanner('success', 'Shard ' + idx + ' removed');
            App.models.load();
          } else {
            var errData = await resp.json().catch(function() { return {}; });
            App.ui.showBanner('error', errData.error ? errData.error.message : 'Failed to remove shard');
          }
        } catch (e) {
          App.ui.showBanner('error', 'Remove failed: ' + e.message);
        }
      } else if (state === 'downloading') {
        App.models.cancelDownload(modelId);
      } else {
        // Try P2P first if peers hold this shard, fall back to HuggingFace
        try {
          if (state === 'peer') {
            // Peers hold this shard — use model acquisition (P2P)
            var p2pResp = await App.authFetch('/api/admin/models/' + encodeURIComponent(modelId) + '/add', { method: 'POST' });
            if (p2pResp.ok) {
              App.ui.showBanner('success', 'Downloading part ' + (idx + 1) + ' from peers');
              App.models.load();
              return;
            }
          }
          // Fallback: HuggingFace
          var srcResp = await App.authFetch('/api/admin/hf/source/' + encodeURIComponent(modelId));
          if (!srcResp.ok) {
            App.ui.showBanner('error', 'No download source found for this model (no HuggingFace source and no peers)');
            return;
          }
          var src = await srcResp.json();
          var dlResp = await App.authFetch('/api/admin/hf/download-shards', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ repo_id: src.repo_id, filename: src.filename, shards: [idx], model_id: modelId }),
          });
          if (dlResp.ok) {
            App.ui.showBanner('success', 'Downloading part ' + (idx + 1) + ' from HuggingFace');
            App.models.load();
          } else {
            var errData2 = await dlResp.json().catch(function() { return {}; });
            App.ui.showBanner('error', errData2.error ? errData2.error.message : 'Download failed');
          }
        } catch (e) {
          App.ui.showBanner('error', 'Download failed: ' + e.message);
        }
      }
    },

    toggleLock: async function() {
      var modelId = this.currentModel;
      var idx = this.currentIndex;
      var newLocked = !this.currentLocked;
      this.hide();
      try {
        var resp = await App.authFetch('/api/admin/models/' + encodeURIComponent(modelId) + '/shards/' + idx + '/lock', {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ locked: newLocked }),
        });
        if (resp.ok) {
          App.ui.showBanner('success', 'Shard ' + idx + (newLocked ? ' locked' : ' unlocked'));
          App.models.load();
        } else {
          App.ui.showBanner('error', 'Failed to update shard lock');
        }
      } catch (e) {
        App.ui.showBanner('error', 'Lock update failed: ' + e.message);
      }
    },

    loadShard: async function() {
      var modelId = this.currentModel;
      var idx = this.currentIndex;
      this.hide();

      try {
        var resp = await App.authFetch('/api/admin/models/' + encodeURIComponent(modelId) + '/shards/' + idx + '/load', { method: 'POST' });
        if (resp.ok) {
          App.notifications.showToast('Loading part ' + (idx + 1) + ' into memory...', 'success');
          App.notifications.logActivity('\u{1F4E5}', U.formatModelDisplayName(modelId) + ': loading part ' + (idx + 1) + ' into memory');
          App.models.load();
        } else {
          var errData = await resp.json().catch(function() { return {}; });
          App.notifications.showToast(errData.error ? errData.error.message : 'Failed to load shard', 'error');
        }
      } catch (e) {
        App.notifications.showToast('Load failed: ' + e.message, 'error');
      }
    },

    unloadShard: async function() {
      var modelId = this.currentModel;
      var idx = this.currentIndex;
      this.hide();

      if (!confirm('Unload part ' + (idx + 1) + ' from memory?\n\nThe file stays on disk. The model worker will restart without this part. Active inference may be briefly interrupted.')) return;

      try {
        // Unload this specific shard — narrows the shard window and restarts the worker.
        // The remaining shards stay loaded; only this one is freed.
        var resp = await App.authFetch('/api/admin/models/' + encodeURIComponent(modelId) + '/shards/' + idx + '/unload', { method: 'POST' });
        if (resp.ok) {
          var name = U.formatModelDisplayName(modelId);
          App.notifications.showToast('Part ' + (idx + 1) + ' of ' + name + ' unloaded from memory', 'success');
          App.notifications.logActivity('\u{1F4A4}', name + ': part ' + (idx + 1) + ' unloaded (manual)');
          App.dashboard._logModelEvent(modelId, '\u{1F4A4}', 'Part ' + (idx + 1) + ' unloaded from memory (manual)');
          App.models.load();
        } else {
          var errData = await resp.json().catch(function() { return {}; });
          App.notifications.showToast(errData.error ? errData.error.message : 'Failed to unload', 'error');
        }
      } catch (e) {
        App.notifications.showToast('Unload failed: ' + e.message, 'error');
      }
    }
  };
})();
