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
    /**
     * Centralized HF shard download request.
     * @param {Object} body - Request body (repo_id, filename, shards?, model_id?, peer_fair_share?)
     * @returns {Promise<{ok: boolean, data: Object, errorMsg: string|null}>}
     */
    downloadShards: async function(body) {
      var resp = await App.authFetch('/api/admin/hf/download-shards', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      var data = await resp.json();
      var errorMsg = null;
      if (!resp.ok) {
        errorMsg = (data.error && data.error.message) || I18n.t('models.download_failed', { error: '' });
      }
      return { ok: resp.ok, data: data, errorMsg: errorMsg };
    },
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
          results.innerHTML = '<div class="empty-state"><p>' + U.escapeHtml(I18n.t('models.search_failed', { error: errBody })) + '</p></div>';
          return;
        }

        var data = await resp.json();

        if (!Array.isArray(data) || data.length === 0) {
          results.innerHTML = '<div class="empty-state"><p>' + U.escapeHtml(I18n.t('models.no_gguf_found', { query: query })) + '</p></div>';
          return;
        }

        // Store data for re-sorting
        App.hf._lastData = data;
        App.hf._renderResults(data);
      } catch (e) {
        loading.classList.add('hidden');
        results.innerHTML = '<div class="empty-state"><p>' + U.escapeHtml(I18n.t('models.search_failed', { error: e.message })) + '</p></div>';
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
          if (repo.downloads) statsHtml += '<span>' + I18n.t('models.downloads_count', { count: repo.downloads.toLocaleString() }) + '</span>';
          if (repo.likes) statsHtml += '<span>' + I18n.t('models.likes_count', { count: repo.likes.toLocaleString() }) + '</span>';
          var shardSizeStr = repo.est_shard_size ? U.formatBytes(repo.est_shard_size) : '';
          var boomerangSizeStr = repo.est_boomerang_size ? U.formatBytes(repo.est_boomerang_size) : '';
          if (repo.fits_boomerang) {
            statsHtml += '<span><span style="color:var(--green)" title="' + U.escapeHtml(I18n.t('models.hf_fit_boomerang', { size: boomerangSizeStr })) + '">&#9989; ' + U.escapeHtml(I18n.t('models.tip_run_local')) + '</span></span>';
          } else if (repo.fits_shard) {
            statsHtml += '<span><span style="color:var(--cyan)" title="' + U.escapeHtml(I18n.t('models.hf_fit_shard', { size: shardSizeStr })) + '">&#128279; ' + U.escapeHtml(I18n.t('models.tip_host_shards')) + '</span></span>';
          } else if (repo.fits_vram === false && variants.length > 0) {
            statsHtml += '<span><span style="color:var(--orange)" title="' + U.escapeHtml(I18n.t('models.hf_exceeds_vram')) + '">&#9888; ' + U.escapeHtml(I18n.t('models.tip_exceeds_vram')) + '</span></span>';
          }
          // Composite score badge
          if (repo.composite_score != null) {
            var scoreColor = repo.composite_score >= 60 ? 'var(--green)' : repo.composite_score >= 30 ? 'var(--yellow)' : 'var(--text-muted)';
            statsHtml += '<span style="color:' + scoreColor + '; font-weight:600" title="' + U.escapeHtml(I18n.t('models.hf_score_breakdown', { quality: (repo.score_breakdown||{}).quality||0, fit: (repo.score_breakdown||{}).fit||0, demand: (repo.score_breakdown||{}).demand||0, size: (repo.score_breakdown||{}).size||0 })) + '">' + I18n.t('models.hf_score_pts', { score: repo.composite_score }) + '</span>';
          }
          card.querySelector('.hf-meta-stats').innerHTML = statsHtml;

          // Network meta
          var replicas = repo.network_replicas || 0;
          var networkHtml = replicas > 0
            ? '<span class="badge-swarm" title="' + U.escapeHtml(I18n.t('models.hf_on_swarm', { count: replicas })) + '">' + U.escapeHtml(I18n.t('models.hf_on_swarm', { count: replicas })) + '</span>'
            : '<span class="badge-new">' + U.escapeHtml(I18n.t('models.badge_new')) + '</span>';
          if (replicas === 0) networkHtml += '<span style="color:var(--green)">&#128176; ' + U.escapeHtml(I18n.t('models.demand_high')) + '</span>';
          else if (replicas <= 2) networkHtml += '<span style="color:var(--yellow)">&#128176; ' + U.escapeHtml(I18n.t('models.demand_medium')) + '</span>';
          else networkHtml += '<span style="color:var(--text-muted)">&#128176; ' + U.escapeHtml(I18n.t('models.well_replicated')) + '</span>';
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
              if (v.quant === recommended) { label += I18n.t('models.hf_recommended'); opt.selected = true; }
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
          App.ui.showBanner('error', I18n.t('models.no_variant_selected'));
          return;
        }

        App.ui.showBanner('info', I18n.t('models.checking'));
        var result = await App.hf.downloadShards({ repo_id: repoId, filename: filename, peer_fair_share: true });
        if (!result.ok) {
          App.ui.showBanner('error', result.errorMsg);
          return;
        }
        if (result.data.status === 'started') {
          App.notifications.showToast(I18n.t('models.download_started'), 'success');
          App.ui.closeModelBrowser();
        } else {
          App.notifications.showToast(result.data.message || I18n.t('models.download_could_not_start'), 'warning');
        }
      } catch (e) {
        App.ui.showBanner('error', I18n.t('models.download_failed', { error: e.message }));
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
            groups.push({ key: 'local', label: I18n.t('models.group_local'), items: localItems });
            S._modelDropdownData = S._modelDropdownData.concat(localItems);
          }
          if (swarmItems.length > 0) {
            groups.push({ key: 'swarm', label: I18n.t('models.group_network'), items: swarmItems });
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
            groups.push({ key: p, label: (PROVIDER_NAMES[p] || p) + I18n.t('models.cloud_suffix'), items: items });
            S._modelDropdownData = S._modelDropdownData.concat(items);
          });
        }

        App.models.renderDropdown(groups, hasAny);

        if (hasAny) {
          var allIds = S._modelDropdownData.map(function(m) { return m.id; });
          var sessionModel = S.currentSessionId && S.sessions[S.currentSessionId] ? S.sessions[S.currentSessionId].model : null;
          var savedModel = null;
          try { savedModel = localStorage.getItem(App.CURRENT_MODEL_KEY); } catch (e) {}
          var preferred = sessionModel || savedModel;
          var found = preferred && allIds.indexOf(preferred) !== -1;
          App.models.selectDropdown(found ? preferred : allIds[0], { silent: true });
        } else {
          S.currentModel = '';
          updateModelDropdownLabel(I18n.t('models.select_model'));
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
        var peerCount = (App.data.cache && App.data.cache.stats) ? (App.data.cache.stats.peers || 0) : 0;
        if (peerCount > 0) {
          list.innerHTML = '<div class="model-dropdown-empty">' +
            I18n.t('models.discovering') +
            '<br><span style="font-size:0.72rem;color:var(--text-muted)">' +
            I18n.t('models.discovering_hint', { count: peerCount }) +
            '</span></div>';
        } else {
          list.innerHTML = '<div class="model-dropdown-empty">' +
            I18n.t('models.no_models') +
            '<br><span style="font-size:0.72rem;color:var(--text-muted)">' +
            I18n.t('models.no_models_hint') +
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
            if (m.context_length || m.context_window) metaParts.push(I18n.t('models.context_abbr', { n: (m.context_length || m.context_window).toLocaleString() }));
            if (m.max_tokens) metaParts.push(I18n.t('models.max_tokens_label', { n: m.max_tokens.toLocaleString() }));
            if (m.pricing) {
              var p = m.pricing;
              if (p.prompt !== undefined) metaParts.push(I18n.t('models.pricing_in', { price: p.prompt }));
              if (p.completion !== undefined) metaParts.push(I18n.t('models.pricing_out', { price: p.completion }));
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
      try { localStorage.setItem(App.CURRENT_MODEL_KEY, modelId); } catch (e) {}

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
          App.notifications.showToast(I18n.t('models.new_session', { model: U.formatModelDisplayName(modelId) }), 'info');
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
        var resp = await App.authFetch(U.modelApiUrl(modelId, 'add'), { method: 'POST' });
        var data = await resp.json();
        if (data.status === 'acquiring') {
          S.activeAcquisitions[modelId] = { started: Date.now() };
          App.dashboard.renderAcquisitionPanel(modelId, null);
        } else {
          App.ui.showBanner('warning', data.message || I18n.t('models.download_unavailable'));
        }
      } catch (e) {
        App.ui.showBanner('error', I18n.t('models.request_failed', { error: e.message }));
      }
    },

    select: function(modelId) {
      App.models.selectDropdown(modelId);
      App.ui.showBanner('success', I18n.t('models.model_selected', { model: modelId }));
      App.models.load();
    },

    cancelDownload: async function(modelId) {
      if (!confirm(I18n.t('actions.confirm_cancel_download', { model: modelId }))) return;
      try {
        var resp = await App.authFetch('/api/admin/downloads/' + encodeURIComponent(modelId) + '/cancel', { method: 'POST' });
        if (resp.ok) {
          App.ui.showBanner('success', I18n.t('models.download_cancelled'));
          var card = document.querySelector('[data-model-id="' + U.cssSafeAttr(modelId) + '"]');
          if (card) {
            var progress = card.querySelector('.dl-progress');
            if (progress) progress.remove();
            card.classList.remove('downloading');
            // Reset any downloading shard rows to missing. Full refresh via
            // loadInitial() below fills in accurate per-shard state.
            card.querySelectorAll('.shard-row[data-state="downloading"]').forEach(function(row) {
              row.setAttribute('data-state', 'missing');
              var pb = row.querySelector('.shard-row-piecebar');
              if (pb) pb.remove();
              var status = row.querySelector('.shard-row-status');
              if (status) status.textContent = I18n.t('shard.row.missing_label');
              var glyph = row.querySelector('.shard-row-state-glyph');
              if (glyph) glyph.textContent = '\u2715';
            });
          }
          delete S.activeAcquisitions[modelId];
          setTimeout(function() { App.dashboard.loadInitial(); }, 1000);
        } else {
          App.ui.showBanner('error', await U.getApiErrorMessage(resp, I18n.t('models.cancel_failed')));
        }
      } catch (e) {
        App.ui.showBanner('error', I18n.t('models.cancel_error', { error: e.message }));
      }
    },

    remove: async function(modelId) {
      if (!confirm(I18n.t('actions.confirm_remove_model', { model: modelId }))) return;
      try {
        var resp = await App.authFetch(U.modelApiUrl(modelId), { method: 'DELETE' });
        if (resp.ok) {
          App.ui.showBanner('success', I18n.t('models.model_removed', { model: modelId }));
          var card = document.querySelector('[data-model-id="' + U.cssSafeAttr(modelId) + '"]');
          if (card) card.remove();
          setTimeout(function() { App.dashboard.loadInitial(); }, 1000);
        } else {
          App.ui.showBanner('error', await U.getApiErrorMessage(resp, I18n.t('models.remove_failed')));
        }
      } catch (e) {
        App.ui.showBanner('error', I18n.t('models.remove_error', { error: e.message }));
      }
    },

    unload: async function(modelId) {
      try {
        var resp = await App.authFetch(U.modelApiUrl(modelId, 'unload'), { method: 'POST' });
        if (resp.ok) {
          var result = await resp.json().catch(function() { return {}; });
          var freedMb = result.estimated_freed_mb || 0;
          var name = result.model_name || U.formatModelDisplayName(modelId);
          var msg = freedMb > 0
            ? I18n.t('models.unloaded_freed', { name: name, freed: U.formatMB(freedMb) })
            : I18n.t('models.unloaded', { name: name });
          App.notifications.showToast(msg, 'success');
          App.models.load();
        } else {
          App.notifications.showToast(await U.getApiErrorMessage(resp, I18n.t('models.unload_failed')), 'error');
        }
      } catch (e) {
        App.notifications.showToast(I18n.t('models.unload_error', { error: e.message }), 'error');
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
          App.authFetch(U.modelApiUrl(modelId, 'auto-manage')),
          App.authFetch(U.modelApiUrl(modelId, 'encrypted-pipeline')),
        ]);
        if (results[0].ok) policy = await results[0].json();
        if (results[1].ok) encStatus = await results[1].json();
      } catch (e) {
        App.ui.showBanner('error', I18n.t('models.policy_load_failed'));
      }

      var encReadyClass = encStatus.ready ? 'text-success' : 'text-warning';
      var encReadyText = encStatus.ready ? I18n.t('models.enc_ready') :
        (!encStatus.has_first_shard ? I18n.t('models.enc_missing_first') + ' ' : '') + (!encStatus.has_last_shard ? I18n.t('models.enc_missing_last') : '');
      var encDisabled = !encStatus.ready ? ' disabled' : '';
      var encOverheadNote = encStatus.shard_count <= 2
        ? '<span class="text-warning" style="font-size:0.65rem">&#9888; ' + U.escapeHtml(I18n.t('models.enc_overhead_local', { count: encStatus.shard_count })) + '</span>'
        : '<span class="text-muted" style="font-size:0.65rem">' + U.escapeHtml(I18n.t('models.enc_overhead')) + '</span>';

      var panel = document.createElement('div');
      panel.className = 'auto-manage-panel';
      panel.innerHTML =
        '<div class="am-row">' +
          '<label><input type="checkbox" id="am-enabled-' + U.safeId(modelId) + '"' + (policy.enabled ? ' checked' : '') + '> ' + U.escapeHtml(I18n.t('models.auto_manage_enabled')) + '</label>' +
        '</div>' +
        '<div class="am-row">' +
          '<label><input type="checkbox" id="am-prune-' + U.safeId(modelId) + '"' + (policy.prune_enabled !== false ? ' checked' : '') + '> ' + U.escapeHtml(I18n.t('models.auto_prune_enabled')) + '</label>' +
        '</div>' +
        '<div class="am-row">' +
          '<label>' + U.escapeHtml(I18n.t('models.max_shards')) + '</label>' +
          '<input type="number" id="am-max-' + U.safeId(modelId) + '" value="' + (policy.max_shards || 0) + '" min="0" step="1">' +
          '<span class="text-muted" style="font-size:0.7rem">' + U.escapeHtml(I18n.t('models.unlimited')) + '</span>' +
        '</div>' +
        '<hr style="margin:0.3rem 0;border-color:var(--border)">' +
        '<div class="am-row" style="flex-direction:column;gap:0.2rem">' +
          '<label><input type="checkbox" id="am-encrypted-' + U.safeId(modelId) + '"' +
            (encStatus.encrypted_pipeline ? ' checked' : '') + encDisabled +
            '> &#128274; ' + U.escapeHtml(I18n.t('models.encrypted_pipeline')) + '</label>' +
          '<span class="' + encReadyClass + '" style="font-size:0.65rem">' + encReadyText + '</span>' +
          encOverheadNote +
        '</div>' +
        '<div class="am-row">' +
          '<button class="btn btn-sm btn-primary" data-am-save="' + U.escapeHtml(modelId) + '">' + U.escapeHtml(I18n.t('actions.save')) + '</button>' +
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
        var amResp = await App.authFetch(U.modelApiUrl(modelId, 'auto-manage'), {
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
          var encResp = await App.authFetch(U.modelApiUrl(modelId, 'encrypted-pipeline'), {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ enabled: encryptedEl.checked }),
          });
          if (!encResp.ok) {
            var encData = await encResp.json().catch(function() { return {}; });
            encErr = U.extractErrorMessage(encData, I18n.t('models.enc_pipeline_save_failed'));
          }
        }

        if (amResp.ok && !encErr) {
          App.ui.showBanner('success', I18n.t('models.policy_saved'));
          var card = document.querySelector('[data-model-id="' + U.cssSafeAttr(modelId) + '"]');
          var panel = card ? card.querySelector('.auto-manage-panel') : null;
          if (panel) panel.remove();
        } else {
          var errMsg = encErr || '';
          if (!amResp.ok) {
            var errData = await amResp.json().catch(function() { return {}; });
            errMsg = U.extractErrorMessage(errData, I18n.t('models.save_failed'));
          }
          App.ui.showBanner('error', errMsg);
        }
      } catch (e) {
        App.ui.showBanner('error', I18n.t('models.save_error', { error: e.message }));
      }
    },

    toggleMetadata: async function(modelId) {
      var panel = document.querySelector('[data-meta-panel="' + U.cssSafeAttr(modelId) + '"]');
      if (!panel) return;
      if (!panel.classList.contains('hidden')) { panel.classList.add('hidden'); return; }
      panel.classList.remove('hidden');
      if (panel.innerHTML) return;

      panel.innerHTML = '<div class="meta-loading"><span class="spinner" style="width:14px;height:14px;border-width:1.5px"></span> ' + U.escapeHtml(I18n.t('models.loading_metadata')) + '</div>';
      try {
        var data = S.metadataCache[modelId];
        if (!data) {
          var resp = await App.authFetch(U.modelApiUrl(modelId, 'metadata'));
          if (!resp.ok) throw new Error('Failed to load metadata');
          data = await resp.json();
          S.metadataCache[modelId] = data;
        }
        renderMetadataPanel(panel, data);
      } catch (e) {
        panel.innerHTML = '<div class="meta-error">' + U.escapeHtml(I18n.t('models.metadata_failed')) + '</div>';
      }
    },

    shutdown: async function() {
      if (!confirm(I18n.t('models.confirm_shutdown'))) return;
      try {
        await App.authFetch('/api/admin/shutdown', { method: 'POST' });
        document.body.innerHTML = '<div style="display:flex;align-items:center;justify-content:center;height:100vh;color:var(--text-muted);font-size:1.2rem">' + U.escapeHtml(I18n.t('models.shutdown_message')) + '</div>';
      } catch (e) {
        App.ui.showBanner('error', I18n.t('models.shutdown_error', { error: e.message }));
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
    var html = '<div class="meta-header">' + U.escapeHtml(I18n.t('models.metadata_header')) + '</div>';
    var g = data.general || {};
    var m = data.model || {};
    var summaryParts = [];
    if (g.architecture) {
      var archTag = '<span class="meta-tag">' + U.escapeHtml(g.architecture) + '</span>';
      if (g.architecture_supported === false) {
        archTag += '<span class="meta-tag" style="background:var(--error-bg,#5c2020);color:var(--error-fg,#ff6b6b)">' + U.escapeHtml(I18n.t('models.meta_unsupported')) + '</span>';
      }
      summaryParts.push(archTag);
    }
    if (g.quantization) summaryParts.push('<span class="meta-tag">' + U.escapeHtml(g.quantization) + '</span>');
    if (m.context_length) summaryParts.push('<span class="meta-tag">ctx ' + m.context_length.toLocaleString() + '</span>');
    if (m.block_count) summaryParts.push('<span class="meta-tag">' + I18n.t('models.meta_layers', { count: m.block_count }) + '</span>');
    if (m.vocab_size) summaryParts.push('<span class="meta-tag">' + I18n.t('models.meta_vocab', { count: m.vocab_size.toLocaleString() }) + '</span>');
    if (summaryParts.length > 0) html += '<div class="meta-summary">' + summaryParts.join('') + '</div>';

    html += '<table class="meta-table"><thead><tr><th colspan="2">' + U.escapeHtml(I18n.t('models.meta_model_params')) + '</th></tr></thead><tbody>';
    var modelFields = [
      [I18n.t('models.meta_context_length'), m.context_length], [I18n.t('models.meta_layers_label'), m.block_count],
      [I18n.t('models.meta_embedding_dim'), m.embedding_length], [I18n.t('models.meta_attention_heads'), m.head_count],
      [I18n.t('models.meta_kv_heads'), m.head_count_kv], [I18n.t('models.meta_rope_dim'), m.rope_dimension_count],
      [I18n.t('models.meta_rope_freq'), m.rope_freq_base], [I18n.t('models.meta_rms_epsilon'), m.layer_norm_rms_epsilon],
      [I18n.t('models.meta_vocab_size'), m.vocab_size],
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
      html += '<table class="meta-table"><thead><tr><th colspan="2">' + U.escapeHtml(I18n.t('models.meta_tokenizer')) + '</th></tr></thead><tbody>';
      [[I18n.t('models.meta_tokenizer_model'), t.model], [I18n.t('models.meta_pre_tokenizer'), t.pre], [I18n.t('models.meta_bos_id'), t.bos_token_id],
       [I18n.t('models.meta_eos_id'), t.eos_token_id], [I18n.t('models.meta_padding_id'), t.padding_token_id]
      ].forEach(function(f) {
        if (f[1] != null) html += '<tr><td class="meta-key">' + U.escapeHtml(f[0]) + '</td><td class="meta-val">' + U.escapeHtml(String(f[1])) + '</td></tr>';
      });
      html += '</tbody></table>';
    }

    var tens = data.tensors || {};
    if (tens.count) html += '<div class="meta-tensor-info">' + I18n.t('models.meta_tensor_info', { count: tens.count, offset: U.formatBytes(tens.data_offset || 0) }) + '</div>';

    var raw = data.raw || [];
    if (raw.length > 0) {
      html += '<details class="meta-raw-details"><summary>' + U.escapeHtml(I18n.t('models.meta_all_keys', { count: raw.length })) + '</summary>';
      html += '<table class="meta-table meta-raw-table"><tbody>';
      raw.forEach(function(r) { html += '<tr><td class="meta-key">' + U.escapeHtml(r.key) + '</td><td class="meta-val">' + U.escapeHtml(r.value) + '</td></tr>'; });
      html += '</tbody></table></details>';
    }
    panel.innerHTML = html;
  }

})();
