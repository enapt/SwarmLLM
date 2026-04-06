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
        toggle.setAttribute('aria-label', I18n.t('settings.toggle_password_aria'));
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

      // Claude subscription toggle
      var csToggle = document.getElementById('claude-subscription-toggle');
      if (csToggle) {
        csToggle.addEventListener('change', async function() {
          try {
            await App.authFetch('/api/admin/providers', {
              method: 'PUT',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ claude_subscription_enabled: csToggle.checked }),
            });
            App.settings.loadProviders();
            App.ui.showBanner('success', I18n.t('settings.claude_subscription_toggled'));
          } catch (e) {
            App.ui.showBanner('error', I18n.t('common.request_failed'));
          }
        });
      }

      // Claude subscription detect button
      var csDetect = document.getElementById('claude-subscription-detect');
      if (csDetect) {
        csDetect.addEventListener('click', async function() {
          var info = document.getElementById('claude-subscription-info');
          var detail = document.getElementById('claude-sub-status-detail');
          if (info) info.textContent = I18n.t('settings.detecting');
          if (detail) detail.style.display = 'none';
          csDetect.disabled = true;
          try {
            var resp = await App.authFetch('/api/admin/claude-subscription/status');
            var data = await resp.json();
            App.settings._updateClaudeSubSteps(data);
            if (data.cli_installed && data.authenticated) {
              var planLabel = data.subscription_type
                ? data.subscription_type.charAt(0).toUpperCase() + data.subscription_type.slice(1)
                : '';
              if (info) info.textContent = I18n.t('settings.claude_sub_ready');
              info.style.color = 'var(--green)';
              if (detail) {
                var parts = [];
                parts.push(I18n.t('settings.cli_version') + ': ' + (data.cli_version || '?'));
                if (planLabel) parts.push(I18n.t('settings.plan') + ': ' + planLabel);
                if (data.rate_limit_tier) parts.push(I18n.t('settings.rate_tier') + ': ' + data.rate_limit_tier);
                detail.innerHTML = parts.join('<span style="margin:0 6px;opacity:0.4">|</span>');
                detail.style.display = '';
                detail.style.color = 'var(--text-secondary)';
              }
            } else if (data.cli_installed && !data.authenticated) {
              if (info) info.textContent = I18n.t('settings.claude_sub_not_logged_in');
              info.style.color = 'var(--orange)';
              if (detail) {
                detail.textContent = I18n.t('settings.claude_sub_login_hint');
                detail.style.display = '';
                detail.style.color = 'var(--orange)';
              }
            } else {
              if (info) info.textContent = I18n.t('settings.cli_not_found');
              info.style.color = 'var(--red)';
              if (detail) {
                detail.textContent = I18n.t('settings.claude_sub_install_hint');
                detail.style.display = '';
                detail.style.color = 'var(--text-secondary)';
              }
            }
          } catch (e) {
            if (info) { info.textContent = I18n.t('common.request_failed'); info.style.color = 'var(--red)'; }
          }
          csDetect.disabled = false;
        });
      }
    },

    _updateClaudeSubSteps: function(data) {
      var checkStyle = 'background:var(--green);color:#fff;border-color:var(--green)';
      var pendingStyle = 'background:var(--bg-tertiary);color:var(--text-primary);border-color:var(--border)';
      var step1 = document.getElementById('claude-sub-step1-icon');
      var step2 = document.getElementById('claude-sub-step2-icon');
      var step3 = document.getElementById('claude-sub-step3-icon');
      var step4 = document.getElementById('claude-sub-step4-icon');
      var toggle = document.getElementById('claude-subscription-toggle');
      if (step1) {
        if (data.cli_installed) { step1.textContent = '\u2713'; step1.style.cssText = checkStyle; }
        else { step1.textContent = '1'; step1.style.cssText = pendingStyle; }
      }
      if (step2) {
        if (data.authenticated) { step2.textContent = '\u2713'; step2.style.cssText = checkStyle; }
        else { step2.textContent = '2'; step2.style.cssText = pendingStyle; }
      }
      if (step3) {
        if (data.cli_installed && data.authenticated) { step3.textContent = '\u2713'; step3.style.cssText = checkStyle; }
        else { step3.textContent = '3'; step3.style.cssText = pendingStyle; }
      }
      if (step4 && toggle) {
        if (toggle.checked) { step4.textContent = '\u2713'; step4.style.cssText = checkStyle; }
        else { step4.textContent = '4'; step4.style.cssText = pendingStyle; }
      }
    },

    load: async function() {
      try {
        var result = await App.data.loadStats();
        var data = result && result.config;
        if (!data) return;
        document.getElementById('settings-contribution').value = data.contribution || 'minimal';
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
        App.ui.showBanner('error', I18n.t('settings.load_failed') + ': ' + (e.message || I18n.t('common.request_failed')));
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
          btn.textContent = I18n.t('actions.copied');
          btn.style.color = 'var(--green)';
          btn.style.borderColor = 'var(--green)';
          setTimeout(function() {
            btn.textContent = I18n.t('actions.copy');
            btn.style.color = '';
            btn.style.borderColor = '';
          }, 2000);
        }
      } catch (e) {
        if (btn) btn.textContent = I18n.t('settings.key_copy_failed');
        setTimeout(function() { if (btn) btn.textContent = I18n.t('actions.copy'); }, 2000);
      }
    },

    loadStorageInfo: async function() {
      try {
        var resp = await App.authFetch('/api/admin/shard-storage');
        var data = await resp.json();
        document.getElementById('settings-storage-used').textContent = U.formatBytes(data.disk_usage_bytes || 0);
        var maxMb = data.auto_manage_max_storage_mb || 0;
        document.getElementById('settings-storage-max').textContent = maxMb > 0 ? U.formatMB(maxMb) : I18n.t('settings.disk_50pct');

        var networkVram = data.pool_vram_mb || 0;
        var localVram = data.local_vram_mb || 0;
        var peerCount = data.peer_count || 0;
        var vramEl = document.getElementById('settings-pool-vram');
        if (vramEl) {
          if (networkVram > 0) {
            vramEl.innerHTML = '<strong>' + U.formatMB(networkVram) + '</strong> ' + U.escapeHtml(I18n.t('settings.network_vram')) +
              ' (' + U.escapeHtml(I18n.t('settings.your_gpu')) + U.formatMB(localVram) + ', ' + peerCount + ' ' + U.escapeHtml(I18n.t('settings.swarm_peers')) + ')';
          } else {
            vramEl.innerHTML = '<span class="text-muted">' + U.escapeHtml(I18n.t('settings.no_gpu')) + '</span>';
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
              var metaText = I18n.t('settings.storage_shards', { local: m.local_shards, total: m.shard_count }) + ' \u00b7 ' + U.formatBytes(m.local_bytes);
              var metaEl = div.querySelector('.storage-model-meta');
              metaEl.textContent = metaText;
              var vramNeeded = m.estimated_vram_mb || 0;
              if (vramNeeded > 0) {
                var vramSpan = document.createElement('span');
                vramSpan.textContent = ' ' + I18n.t('settings.vram_label', { size: U.formatMB(vramNeeded) });
                var fits = networkVram > 0 && vramNeeded <= networkVram;
                var tooLarge = networkVram > 0 && vramNeeded > networkVram;
                if (fits) vramSpan.style.color = 'var(--green)';
                else if (tooLarge) { vramSpan.style.color = 'var(--red)'; vramSpan.title = I18n.t('settings.exceeds_vram', { size: U.formatMB(networkVram) }); }
                else vramSpan.className = 'text-muted';
                metaEl.appendChild(vramSpan);
              }
              modelsDiv.appendChild(div);
            }
          });
          if (modelsDiv.children.length === 0) {
            modelsDiv.innerHTML = '<span class="text-muted">' + U.escapeHtml(I18n.t('settings.no_local_shards')) + '</span>';
          }
        } else {
          modelsDiv.innerHTML = '<span class="text-muted">' + U.escapeHtml(I18n.t('settings.no_models')) + '</span>';
        }
      } catch (e) {
        App.ui.showBanner('error', I18n.t('settings.storage_load_failed'));
      }
    },

    loadProviders: async function() {
      try {
        var data = await App.data.loadProviders();
        data = data || {};
        if (data.providers) {
          var anyConfigured = false;
          data.providers.forEach(function(p) {
            if (p.configured) anyConfigured = true;
            var badge = document.getElementById('provider-status-' + p.name);
            if (badge) {
              if (p.configured && p.source === 'env') {
                badge.textContent = I18n.t('settings.badge_from_env');
                badge.className = 'badge provider-badge-active';
              } else if (p.configured) {
                badge.textContent = I18n.t('settings.badge_active');
                badge.className = 'badge provider-badge-active';
              } else {
                badge.textContent = I18n.t('settings.badge_not_set');
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
        // Claude subscription UI
        if (data.claude_subscription !== undefined) {
          var card = document.getElementById('claude-subscription-card');
          if (card) card.style.display = '';
          var toggle = document.getElementById('claude-subscription-toggle');
          if (toggle) toggle.checked = !!(data.claude_subscription && data.claude_subscription.enabled);
          var badge = document.getElementById('claude-subscription-status');
          if (badge && data.claude_subscription && data.claude_subscription.enabled) {
            badge.textContent = I18n.t('settings.badge_active');
            badge.className = 'badge provider-badge-active';
            if (card) card.classList.add('provider-active');
          } else if (badge) {
            badge.textContent = I18n.t('settings.claude_subscription_disabled');
            badge.className = 'badge';
            if (card) card.classList.remove('provider-active');
          }
        }
      } catch (e) {
        App.ui.showBanner('error', I18n.t('settings.providers_load_failed'));
      }
    },

    saveProviders: async function() {
      var providerNames = ['anthropic', 'openai', 'deepseek', 'mistral', 'groq', 'nvidia_nim', 'cerebras', 'sambanova', 'fireworks', 'together', 'deepinfra', 'moonshot'];
      var keys = {};
      providerNames.forEach(function(name) {
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
        providerNames.forEach(function(name) {
          var input = document.getElementById('provider-key-' + name);
          if (input) input.value = '';
        });
        App.settings.loadProviders();
        App.models.load();
        App.modeIndicator.load();
        App.providerHealth.startHealthPolling();
        App.ui.showBanner('success', I18n.t('settings.providers_saved'));
      } catch (e) {
        App.ui.showBanner('error', I18n.t('settings.providers_save_failed') + ': ' + (e.message || I18n.t('common.request_failed')));
      }
    },

    testProvider: async function(name) {
      var input = document.getElementById('provider-key-' + name);
      var badge = document.getElementById('provider-status-' + name);
      if (!input) return;
      var key = input.value;
      if (!key) {
        App.ui.showBanner('error', I18n.t('settings.enter_key_first'));
        return;
      }
      badge.textContent = I18n.t('settings.badge_testing');
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
          badge.textContent = I18n.t('settings.badge_active');
          badge.className = 'badge provider-badge-active';
          App.ui.showBanner('success', I18n.t('settings.key_verified', { name: name }));
          var testCard = badge.closest('.provider-card');
          if (testCard) testCard.classList.add('provider-active');
          App.models.load();
          App.modeIndicator.load();
        } else {
          var err = await testResp.text();
          var friendlyErr = err;
          try { var ej = JSON.parse(err); friendlyErr = (ej.error && ej.error.message) || err; } catch(pe) {}
          if (friendlyErr.length > 200) friendlyErr = friendlyErr.substring(0, 200) + '\u2026';
          badge.textContent = I18n.t('settings.badge_failed');
          badge.className = 'badge badge-error';
          App.ui.showBanner('error', I18n.t('settings.key_test_failed', { name: name, error: friendlyErr }));
        }
        input.value = '';
      } catch (e) {
        badge.textContent = I18n.t('settings.badge_error');
        badge.className = 'badge badge-error';
        App.ui.showBanner('error', I18n.t('settings.key_test_failed', { name: name, error: e.message }));
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
          // Save ancillary settings only if main config save succeeded
          var healthIntervalEl = document.getElementById('settings-health-interval');
          if (healthIntervalEl) {
            try { localStorage.setItem(App.HEALTH_INTERVAL_KEY, healthIntervalEl.value); } catch(e) {}
            App.providerHealth.startHealthPolling();
          }
          await App.identity.saveNickname();
          await App.settings.saveProviders();

          App.ui.showBanner('success', I18n.t('settings.saved'));
          App.ui.closeSettings();
          App.dashboard.loadInitial();
        } else {
          var errMsg = await U.getApiErrorMessage(resp, I18n.t('settings.save_failed'));
          App.ui.showBanner('error', errMsg);
        }
      } catch (e) {
        App.ui.showBanner('error', I18n.t('settings.save_error', { error: e.message }));
      }

      if (saveBtn) { saveBtn.disabled = false; saveBtn.textContent = I18n.t('actions.save_settings'); }
    }
  };

})();
