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
        toggle.textContent = 'Show';
        toggle.setAttribute('aria-label', 'Toggle password visibility');
        toggle.addEventListener('click', function() {
          var isPass = input.type === 'password';
          input.type = isPass ? 'text' : 'password';
          toggle.textContent = isPass ? 'Hide' : 'Show';
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
        var resp = await fetch('/api/admin/config');
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
        var resp = await fetch('/api/admin/api-key');
        if (resp.ok) {
          var data = await resp.json();
          var key = data.api_key || '';
          App.settings._apiKeyFull = key;
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
      if (!App.settings._apiKeyFull) return;
      try {
        await navigator.clipboard.writeText(App.settings._apiKeyFull);
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
        document.getElementById('settings-storage-used').textContent = U.formatBytes(data.disk_usage_bytes || 0);
        var maxMb = data.auto_manage_max_storage_mb || 0;
        document.getElementById('settings-storage-max').textContent = maxMb > 0 ? U.formatMB(maxMb) : '50% of disk limit';

        var poolVram = data.pool_vram_mb || 0;
        var localVram = data.local_vram_mb || 0;
        var peerCount = data.peer_count || 0;
        var poolEl = document.getElementById('settings-pool-vram');
        if (poolEl) {
          if (poolVram > 0) {
            poolEl.innerHTML = '<strong>' + U.formatMB(poolVram) + '</strong> total VRAM' +
              ' (local: ' + U.formatMB(localVram) + ', ' + peerCount + ' peer' + (peerCount !== 1 ? 's' : '') + ')';
          } else {
            poolEl.innerHTML = '<span class="text-muted">No GPU detected</span>';
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
                var fits = poolVram > 0 && vramNeeded <= poolVram;
                var tooLarge = poolVram > 0 && vramNeeded > poolVram;
                if (fits) vramSpan.style.color = 'var(--green)';
                else if (tooLarge) { vramSpan.style.color = 'var(--red)'; vramSpan.title = 'Exceeds pool VRAM (' + U.formatMB(poolVram) + ')'; }
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

  // ========================================================================
  // Setup Wizard
  // ========================================================================
  App.setup = {
    currentStep: 1,
    hwData: null,
    _savedProvider: null,

    init: function() {
      if (localStorage.getItem(App.SETUP_DONE_KEY) === 'true') return;
      document.getElementById('setup-modal').classList.remove('hidden');
      App.setup.detectHardware();

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
        App.setup.hwData = data.hardware || {};
        document.getElementById('hw-gpu').textContent = App.setup.hwData.gpu_name || 'No GPU (CPU mode)';
        document.getElementById('hw-vram').textContent = App.setup.hwData.gpu_vram_mb ? U.formatMB(App.setup.hwData.gpu_vram_mb) : 'N/A';
        document.getElementById('hw-ram').textContent = U.formatMB(App.setup.hwData.total_ram_mb || 0);
        document.getElementById('hw-disk').textContent = U.formatMB(App.setup.hwData.available_disk_mb || 0);
      } catch (e) {
        document.getElementById('hw-gpu').textContent = 'Detection failed';
        App.setup.hwData = {};
      }
      document.getElementById('hw-loading').classList.add('hidden');
      document.getElementById('hw-results').classList.remove('hidden');
    },

    nextStep: function() {
      if (App.setup.currentStep === 4) {
        App.setup.submit();
        return;
      }
      App.setup.currentStep++;
      App.setup.updateUI();
      if (App.setup.currentStep === 3) App.setup.loadModelSelection();
      if (App.setup.currentStep === 4) App.setup.populateSummary();
    },

    prevStep: function() {
      if (App.setup.currentStep > 1) {
        App.setup.currentStep--;
        App.setup.updateUI();
      }
    },

    updateUI: function() {
      for (var i = 1; i <= 4; i++) {
        var body = document.getElementById('step-' + i);
        var indicator = document.querySelector('[data-step="' + i + '"]');
        if (i === App.setup.currentStep) {
          body.classList.remove('hidden');
          indicator.classList.add('active');
          indicator.classList.remove('done');
          indicator.setAttribute('aria-selected', 'true');
        } else if (i < App.setup.currentStep) {
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
      connectors.forEach(function(c, idx) { c.classList.toggle('done', idx + 1 < App.setup.currentStep); });
      document.getElementById('btn-prev').classList.toggle('hidden', App.setup.currentStep === 1);
      document.getElementById('btn-next').textContent = App.setup.currentStep === 4 ? 'Start SwarmLLM' : 'Continue';
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
            document.getElementById('setup-modal').style.display = 'none';
            App.ui.openSettings();
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
            var size = m.total_size_bytes ? U.formatBytes(m.total_size_bytes) : '';
            var status = m.status === 'loaded' ? '<span class="text-green" style="font-size:0.8rem">Loaded</span>' : '<span class="text-muted" style="font-size:0.8rem">' + (m.status || 'available') + '</span>';
            div.innerHTML = '<div style="display:flex;justify-content:space-between;align-items:center"><strong style="font-size:0.9rem">' + U.escapeHtml(name) + '</strong>' + status + '</div>' +
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
      document.getElementById('summary-contribution').textContent = U.capitalize(levels[val]);
      document.getElementById('summary-gpu').textContent = App.setup.hwData && App.setup.hwData.gpu_name ? App.setup.hwData.gpu_name : 'CPU only';
      document.getElementById('summary-ram').textContent = U.formatMB(App.setup.hwData ? App.setup.hwData.total_ram_mb || 0 : 0);
      document.getElementById('summary-disk').textContent = U.formatMB(App.setup.hwData ? App.setup.hwData.available_disk_mb || 0 : 0);
      var autoManage = document.getElementById('setup-auto-manage').checked;
      document.getElementById('summary-auto-manage').textContent = autoManage ? 'Enabled' : 'Disabled';
      var provNames = {openai:'OpenAI',deepseek:'DeepSeek',groq:'Groq',nvidia_nim:'NVIDIA NIM',cerebras:'Cerebras',sambanova:'SambaNova',anthropic:'Anthropic',mistral:'Mistral',fireworks:'Fireworks',together:'Together',deepinfra:'DeepInfra'};
      document.getElementById('summary-provider').textContent = App.setup._savedProvider ? provNames[App.setup._savedProvider] || App.setup._savedProvider : 'None (can add later)';
      document.getElementById('summary-models').textContent = 'Default configuration';
    },

    submit: async function() {
      var levels = ['minimal', 'moderate', 'maximum'];
      var level = levels[parseInt(document.getElementById('contribution-slider').value, 10)];
      var autoManage = document.getElementById('setup-auto-manage').checked;
      try {
        var resp = await App.authFetch('/api/admin/config', {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            contribution: level,
            auto_manage_shards: autoManage,
          }),
        });
        if (!resp.ok) {
          App.ui.showBanner('error', 'Setup failed \u2014 could not save configuration');
          return;
        }
      } catch (e) {
        App.ui.showBanner('error', 'Setup failed: ' + (e.message || 'network error'));
        return;
      }
      var nick = (document.getElementById('setup-nickname').value || '').trim();
      if (nick) {
        try {
          await App.authFetch('/api/identity/nickname', {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ nickname: nick, visibility: 'nickname' }),
          });
        } catch (e) {}
      }
      localStorage.setItem(App.SETUP_DONE_KEY, 'true');
      try {
        await App.authFetch('/api/admin/config', {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ setup_done: true }),
        });
      } catch (e) {}
      document.getElementById('setup-modal').classList.add('hidden');
      App.ui.showBanner('success', 'Setup complete! Welcome to SwarmLLM.');
    },

    finish: function() {
      localStorage.setItem(App.SETUP_DONE_KEY, 'true');
      document.getElementById('setup-modal').classList.add('hidden');
      App.ui.showBanner('info', 'Setup skipped \u2014 you can configure everything in Settings.');
    }
  };
})();
