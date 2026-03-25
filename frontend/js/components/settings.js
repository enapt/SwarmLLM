'use strict';

// ============================================================================
// SwarmLLM — Settings + Setup Wizard Component
// ============================================================================

(function() {
  var S = App.state;
  var U = App.utils;

  // Pre-initialize fields that data.js references
  App.settings = {
    _apiKeyFull: '',
    _apiKeyPromise: null,

    init: function() {
      var autoSelect = document.getElementById('settings-auto-shards');
      if (autoSelect) {
        autoSelect.addEventListener('change', function() {
          var isOn = this.value === 'on';
          document.getElementById('settings-auto-manage-storage-group').style.display = isOn ? '' : 'none';
          document.getElementById('settings-storage-info').classList.toggle('hidden', !isOn);
          if (isOn) App.settings.loadStorageInfo();
        });
      }
      var healthIntervalEl = document.getElementById('settings-health-interval');
      if (healthIntervalEl) {
        try { var saved = localStorage.getItem(App.HEALTH_INTERVAL_KEY); if (saved) healthIntervalEl.value = saved; } catch(e) {}
      }

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
        toggle.textContent = I18n.t('settings.show_password');
        toggle.setAttribute('aria-label', 'Toggle password visibility');
        toggle.addEventListener('click', function() {
          var isPass = input.type === 'password';
          input.type = isPass ? 'text' : 'password';
          toggle.textContent = isPass ? I18n.t('settings.hide_password') : I18n.t('settings.show_password');
        });
        wrap.appendChild(toggle);
      });

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
        var resp = await App.authFetch('/api/admin/config');
        if (!resp.ok) return;
        var data = await resp.json();
        document.getElementById('settings-contribution').value = data.contribution || 'moderate';
        document.getElementById('settings-max-requests').value = data.max_concurrent_requests || 10;
        document.getElementById('settings-bandwidth').value = data.max_bandwidth_mbps || 0;
        document.getElementById('settings-disk').value = data.max_disk_mb || 50000;
        var autoManage = data.auto_manage_shards ? 'on' : 'off';
        document.getElementById('settings-auto-shards').value = autoManage;
        document.getElementById('settings-auto-manage-storage').value = data.auto_manage_max_storage_mb || 0;
        var isOn = autoManage === 'on';
        document.getElementById('settings-auto-manage-storage-group').style.display = isOn ? '' : 'none';
        document.getElementById('settings-storage-info').classList.toggle('hidden', !isOn);
        if (isOn) App.settings.loadStorageInfo();
      } catch (e) {
        App.ui.showBanner('error', 'Failed to load settings: ' + (e.message || 'network error'));
      }
      App.settings._apiKeyPromise = App.settings.loadApiKey();
      App.settings.loadProviders();
    },

    loadApiKey: async function() {
      var keyEl = document.getElementById('settings-api-key');
      if (!keyEl) return;
      try {
        var resp = await App.authFetch('/api/admin/api-key');
        if (resp.ok) {
          var data = await resp.json();
          var key = data.api_key || '';
          App.settings._apiKeyFull = key;
          keyEl.value = key ? key.substring(0, 4) + '****' + key.substring(key.length - 4) : I18n.t('settings.no_api_key');
        } else {
          keyEl.value = I18n.t('settings.key_unavailable');
        }
      } catch (e) {
        keyEl.value = I18n.t('settings.key_error');
      }
    },

    copyApiKey: async function() {
      var btn = document.getElementById('btn-copy-api-key');
      if (!App.settings._apiKeyFull) return;
      try {
        await navigator.clipboard.writeText(App.settings._apiKeyFull);
        if (btn) {
          btn.textContent = I18n.t('settings.key_copied');
          btn.style.color = 'var(--green)';
          btn.style.borderColor = 'var(--green)';
          setTimeout(function() {
            btn.textContent = I18n.t('settings.key_copy');
            btn.style.color = '';
            btn.style.borderColor = '';
          }, 2000);
        }
      } catch (e) {
        if (btn) btn.textContent = I18n.t('settings.key_copy_failed');
        setTimeout(function() { if (btn) btn.textContent = I18n.t('settings.key_copy'); }, 2000);
      }
    },

    loadStorageInfo: async function() {
      try {
        var resp = await App.authFetch('/api/admin/shard-storage');
        var data = await resp.json();
        document.getElementById('settings-storage-used').textContent = U.formatBytes(data.disk_usage_bytes || 0);
        var maxMb = data.auto_manage_max_storage_mb || 0;
        document.getElementById('settings-storage-max').textContent = maxMb > 0 ? U.formatMB(maxMb) : '50% of disk limit';

        var networkVram = data.pool_vram_mb || 0;
        var localVram = data.local_vram_mb || 0;
        var peerCount = data.peer_count || 0;
        var vramEl = document.getElementById('settings-pool-vram');
        if (vramEl) {
          if (networkVram > 0) {
            vramEl.innerHTML = '<strong>' + U.formatMB(networkVram) + '</strong> swarm network VRAM' +
              ' (your GPU: ' + U.formatMB(localVram) + ', ' + peerCount + ' swarm peer' + (peerCount !== 1 ? 's' : '') + ')';
          } else {
            vramEl.innerHTML = '<span class="text-muted">No GPU detected</span>';
          }
        }

        var modelsDiv = document.getElementById('settings-storage-models');
        modelsDiv.innerHTML = '';
        var storageTmpl = document.getElementById('tmpl-storage-model-row');
        if (data.models && data.models.length > 0) {
          data.models.forEach(function(m) {
            if (m.local_shards > 0) {
              var div = storageTmpl.content.cloneNode(true).firstElementChild;
              div.querySelector('.storage-model-name').textContent = m.name || m.id;
              var metaText = m.local_shards + '/' + m.shard_count + ' shards \u00b7 ' + U.formatBytes(m.local_bytes);
              var metaEl = div.querySelector('.storage-model-meta');
              metaEl.textContent = metaText;
              var vramNeeded = m.estimated_vram_mb || 0;
              if (vramNeeded > 0) {
                var vramSpan = document.createElement('span');
                vramSpan.textContent = ' ' + U.formatMB(vramNeeded) + ' VRAM';
                var fits = networkVram > 0 && vramNeeded <= networkVram;
                var tooLarge = networkVram > 0 && vramNeeded > networkVram;
                if (fits) vramSpan.style.color = 'var(--green)';
                else if (tooLarge) { vramSpan.style.color = 'var(--red)'; vramSpan.title = 'Exceeds network VRAM (' + U.formatMB(networkVram) + ')'; }
                else vramSpan.className = 'text-muted';
                metaEl.appendChild(vramSpan);
              }
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
        App.ui.showBanner('error', 'Failed to load storage info');
      }
    },

    loadProviders: async function() {
      try {
        var resp = await App.authFetch('/api/admin/providers');
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
            var card = badge && badge.closest('.provider-card');
            if (card) {
              if (p.configured) {
                card.classList.add('provider-active');
              } else {
                card.classList.remove('provider-active');
              }
            }
          });
          if (!anyConfigured) {
            var section = document.getElementById('settings-providers-section');
            if (section) section.open = true;
          }
        }
        if (data.key_source) {
          var sel = document.getElementById('provider-key-source');
          if (sel) sel.value = data.key_source;
        }
      } catch (e) {
        App.ui.showBanner('error', 'Failed to load provider status');
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
        await App.authFetch('/api/admin/providers', {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(keys),
        });
        ['anthropic', 'openai', 'deepseek', 'mistral', 'groq', 'nvidia_nim', 'cerebras', 'sambanova', 'fireworks', 'together', 'deepinfra', 'moonshot'].forEach(function(name) {
          var input = document.getElementById('provider-key-' + name);
          if (input) input.value = '';
        });
        App.settings.loadProviders();
        App.models.load();
        App.modeIndicator.load();
        App.providerHealth.startHealthPolling();
        App.ui.showBanner('success', 'Provider keys saved');
      } catch (e) {
        App.ui.showBanner('error', 'Failed to save provider keys: ' + (e.message || 'network error'));
      }
    },

    testProvider: async function(name) {
      var input = document.getElementById('provider-key-' + name);
      var badge = document.getElementById('provider-status-' + name);
      if (!input) return;
      var key = input.value;
      if (!key) {
        App.ui.showBanner('error', 'Enter an API key first');
        return;
      }
      badge.textContent = 'Testing...';
      badge.className = 'badge badge-testing';
      try {
        var saveBody = {};
        saveBody[name + '_key'] = key;
        await App.authFetch('/api/admin/providers', {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(saveBody),
        });
        var testResp;
        if (name === 'anthropic') {
          testResp = await App.authFetch('/v1/messages', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ model: 'claude-haiku-4-5-20251001', max_tokens: 1, messages: [{ role: 'user', content: 'hi' }] }),
          });
        } else {
          var modelMap = { openai: 'gpt-4o-mini', deepseek: 'deepseek-chat', mistral: 'mistral-small-latest', groq: 'llama-3.1-8b-instant', nvidia_nim: 'meta/llama-3.1-8b-instruct', cerebras: 'cerebras:llama-3.1-8b', sambanova: 'sambanova:Meta-Llama-3.3-70B-Instruct', fireworks: 'accounts/fireworks/models/llama-v3p3-70b-instruct', together: 'together:meta-llama/Llama-3.3-70B-Instruct-Turbo', deepinfra: 'deepinfra:meta-llama/Llama-3.3-70B-Instruct', moonshot: 'moonshot-v1-8k' };
          testResp = await App.authFetch('/v1/chat/completions', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ model: modelMap[name] || name + '-test', max_tokens: 1, messages: [{ role: 'user', content: 'hi' }] }),
          });
        }
        if (testResp.ok) {
          badge.textContent = '\u2713 Active';
          badge.className = 'badge provider-badge-active';
          App.ui.showBanner('success', name + ' API key verified');
          var testCard = badge.closest('.provider-card');
          if (testCard) testCard.classList.add('provider-active');
          App.models.load();
          App.modeIndicator.load();
        } else {
          var err = await testResp.text();
          var friendlyErr = err;
          try { var ej = JSON.parse(err); friendlyErr = (ej.error && ej.error.message) || err; } catch(pe) {}
          if (friendlyErr.length > 200) friendlyErr = friendlyErr.substring(0, 200) + '\u2026';
          badge.textContent = '\u2717 Failed';
          badge.className = 'badge badge-error';
          App.ui.showBanner('error', name + ' test failed: ' + friendlyErr);
        }
        input.value = '';
      } catch (e) {
        badge.textContent = '\u2717 Error';
        badge.className = 'badge badge-error';
        App.ui.showBanner('error', name + ' test failed: ' + e.message);
      }
    },

    save: async function() {
      var saveBtn = document.getElementById('btn-save-settings');
      if (saveBtn) { saveBtn.disabled = true; saveBtn.textContent = I18n.t('actions.saving'); }

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
        var resp = await App.authFetch('/api/admin/config', {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(config),
        });
        if (resp.ok) {
          App.ui.showBanner('success', 'Settings saved');
          App.ui.closeSettings();
          // Refresh dashboard to reflect new config immediately
          App.dashboard.loadInitial();
        } else {
          App.ui.showBanner('error', 'Failed to save settings');
        }
      } catch (e) {
        App.ui.showBanner('error', 'Error: ' + e.message);
      }

      var healthIntervalEl = document.getElementById('settings-health-interval');
      if (healthIntervalEl) {
        try { localStorage.setItem(App.HEALTH_INTERVAL_KEY, healthIntervalEl.value); } catch(e) {}
        App.providerHealth.startHealthPolling();
      }

      await App.identity.saveNickname();
      await App.settings.saveProviders();

      if (saveBtn) { saveBtn.disabled = false; saveBtn.textContent = I18n.t('actions.save_settings'); }
    }
  };

})();
