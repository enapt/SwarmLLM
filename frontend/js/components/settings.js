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

    _formatSlider: function(fmt, v) {
      v = parseInt(v, 10) || 0;
      switch (fmt) {
        case 'int': return String(v);
        case 'mbps': return v === 0 ? I18n.t('settings.slider_unlimited') : v + ' Mbps';
        case 'gb': return (v / 1000).toFixed(v < 10000 ? 1 : 0) + ' GB';
        case 'gb-auto': return v === 0 ? I18n.t('settings.slider_auto') : (v / 1000).toFixed(v < 10000 ? 1 : 0) + ' GB';
        case 'sec-off': return v === 0 ? I18n.t('settings.slider_off') : v + ' s';
        default: return String(v);
      }
    },
    _wireSliders: function() {
      var self = this;
      document.querySelectorAll('input.slider[data-slider-format]').forEach(function(el) {
        var out = document.getElementById(el.id + '-out');
        if (!out) return;
        var fmt = el.getAttribute('data-slider-format');
        var update = function() { out.textContent = self._formatSlider(fmt, el.value); };
        el.addEventListener('input', update);
        el.addEventListener('change', update);
        // Track via property so loadSettings can trigger re-render after value change
        el._sliderUpdate = update;
        update();
      });
    },
    _wireSegmented: function() {
      document.querySelectorAll('.segmented[data-bound-select]').forEach(function(seg) {
        var sel = document.getElementById(seg.getAttribute('data-bound-select'));
        if (!sel) return;
        var sync = function() {
          var v = sel.value;
          seg.querySelectorAll('.segmented-btn').forEach(function(b) {
            b.classList.toggle('active', b.getAttribute('data-value') === v);
          });
        };
        seg.querySelectorAll('.segmented-btn').forEach(function(btn) {
          btn.addEventListener('click', function() {
            sel.value = btn.getAttribute('data-value');
            sel.dispatchEvent(new Event('change'));
            sync();
          });
        });
        sel.addEventListener('change', sync);
        seg._segSync = sync;
        sync();
      });
    },

    renderHwModeNote: function(el, isGpu) {
      if (!el) return;
      el.hidden = false;
      el.classList.remove('mode-cpu', 'mode-gpu');
      el.classList.add(isGpu ? 'mode-gpu' : 'mode-cpu');
      var icon = isGpu ? '◈' : '≡';
      var title = I18n.t(isGpu ? 'hw.mode_note_gpu_title' : 'hw.mode_note_cpu_title');
      var body = I18n.t(isGpu ? 'hw.mode_note_gpu_body' : 'hw.mode_note_cpu_body');
      el.innerHTML = '<span class="hw-mode-note-icon" aria-hidden="true">' + U.escapeHtml(icon) + '</span>' +
        '<div class="hw-mode-note-body"><strong>' + U.escapeHtml(title) + '</strong>' + U.escapeHtml(body) + '</div>';
    },
    _applyHwMode: function(hw) {
      // Fallback: dashboard caches flag on App.state._gpuInference
      var isGpu;
      if (hw && typeof hw.gpu_inference === 'boolean') isGpu = hw.gpu_inference;
      else isGpu = !!(App.state && App.state._gpuInference);
      var backend = (hw && hw.inference_backend) || 'GPU';
      var unit = isGpu ? 'GPU' : 'CPU';

      var badge = document.getElementById('settings-mode-badge');
      if (badge) {
        if (isGpu) {
          badge.textContent = I18n.t('hw.gpu_mode_label', { backend: backend });
          badge.className = 'node-mode-badge node-mode-gpu';
          badge.title = I18n.t('hw.gpu_mode_tip');
        } else {
          badge.textContent = I18n.t('hw.mode_cpu');
          badge.className = 'node-mode-badge node-mode-cpu';
          badge.title = I18n.t('hw.cpu_mode_tip');
        }
      }

      // Update contribution sub-labels to show only the relevant resource
      var subs = { minimal: '≤ 25%', moderate: '≤ 50%', maximum: '≤ 75%+' };
      document.querySelectorAll('.segmented[data-bound-select="settings-contribution"] .segmented-btn').forEach(function(btn) {
        var v = btn.getAttribute('data-value');
        var sub = btn.querySelector('.segmented-sub');
        if (sub && subs[v]) sub.textContent = subs[v] + ' ' + unit;
      });

      // Update contribution hint + compute memory label
      var hint = document.querySelector('#settings-contribution ~ .field-hint, .form-group .field-hint[data-i18n="settings.contribution_hint"]');
      if (hint) {
        hint.textContent = I18n.t(isGpu ? 'settings.contribution_hint_gpu' : 'settings.contribution_hint_cpu');
      }
      this.renderHwModeNote(document.getElementById('settings-mode-note'), isGpu);

      var memLabel = document.querySelector('[data-i18n="settings.swarm_compute_memory"]');
      if (memLabel) {
        memLabel.textContent = I18n.t(isGpu ? 'settings.swarm_compute_memory_gpu' : 'settings.swarm_compute_memory_cpu');
      }
    },

    init: function() {
      this._wireSliders();
      this._wireSegmented();
      this._applyHwMode(null);
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
            // User-triggered refresh: clear in-flight dedup so we get fresh data
            App.data.invalidateDedup('claudeSubStatus');
            var data = await App.data.loadClaudeSubStatus();
            if (!data) { throw new Error('No response'); }
            App.settings._updateClaudeSubSteps(data);
            if (data.cli_installed && data.authenticated) {
              var planLabel = data.subscription_type
                ? data.subscription_type.charAt(0).toUpperCase() + data.subscription_type.slice(1)
                : '';
              if (info) info.textContent = I18n.t('settings.claude_sub_ready');
              info.style.color = 'var(--green)';
              if (detail) {
                var parts = [
                  I18n.t('settings.cli_version') + ': ' + (data.cli_version || '?'),
                ];
                if (planLabel) parts.push(I18n.t('settings.plan') + ': ' + planLabel);
                if (data.rate_limit_tier) parts.push(I18n.t('settings.rate_tier') + ': ' + data.rate_limit_tier);
                detail.textContent = '';
                parts.forEach(function(p, i) {
                  if (i > 0) {
                    var sep = document.createElement('span');
                    sep.style.cssText = 'margin:0 6px;opacity:0.4';
                    sep.textContent = '|';
                    detail.appendChild(sep);
                  }
                  detail.appendChild(document.createTextNode(p));
                });
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
      function setStep(id, num, done) {
        var el = document.getElementById(id);
        if (!el) return;
        el.textContent = done ? '\u2713' : String(num);
        el.style.cssText = done ? checkStyle : pendingStyle;
      }
      var toggle = document.getElementById('claude-subscription-toggle');
      setStep('claude-sub-step1-icon', 1, data.cli_installed);
      setStep('claude-sub-step2-icon', 2, data.authenticated);
      setStep('claude-sub-step3-icon', 3, data.cli_installed && data.authenticated);
      if (toggle) setStep('claude-sub-step4-icon', 4, toggle.checked);
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
        if (App.autoManageStatus) App.autoManageStatus.setEnabled(isOn);
        // Refresh slider outputs + segmented buttons now that values are loaded
        document.querySelectorAll('input.slider[data-slider-format]').forEach(function(el) { if (el._sliderUpdate) el._sliderUpdate(); });
        document.querySelectorAll('.segmented[data-bound-select]').forEach(function(seg) { if (seg._segSync) seg._segSync(); });
        App.settings._applyHwMode(result && result.hardware);
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
        // Bootstrap path: include the per-page nonce the daemon embedded
        // in the served HTML so the loopback gate in src/api/middleware.rs
        // accepts the request without a Bearer (we don't have one yet —
        // this fetch is how we obtain it). Single-use, 60s TTL. The nonce
        // is stripped from the meta tag after consumption to avoid replay
        // from any code that re-reads it.
        var nonceEl = document.querySelector('meta[name="bootstrap-nonce"]');
        var nonce = nonceEl ? nonceEl.getAttribute('content') : '';
        var headers = {};
        if (nonce && nonce !== '__SWARMLLM_BOOTSTRAP_NONCE__') {
          headers['X-Dashboard-Nonce'] = nonce;
          if (nonceEl) nonceEl.setAttribute('content', '');
        }
        var resp = await App.authFetch('/api/admin/api-key', { headers: headers });
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
      await U.copyToClipboard(App.settings._apiKeyFull, {
        btn: btn,
        successLabel: I18n.t('actions.copied'),
        failLabel: I18n.t('settings.key_copy_failed'),
        resetLabel: I18n.t('actions.copy'),
      });
    },

    loadStorageInfo: async function() {
      try {
        var resp = await App.authFetch('/api/admin/shard-storage');
        var data = await resp.json();
        document.getElementById('settings-storage-used').textContent = U.formatBytes(data.disk_usage_bytes || 0);
        var maxMb = data.auto_manage_max_storage_mb || 0;
        document.getElementById('settings-storage-max').textContent = maxMb > 0 ? U.formatMB(maxMb) : I18n.t('settings.disk_50pct');

        // R110: render the stacked-bar from the dedicated breakdown
        // endpoint. Best-effort — we don't fail the whole storage panel
        // if the new endpoint isn't reachable (e.g. older daemon during
        // upgrade roll-out).
        try {
          var bResp = await App.authFetch('/api/admin/storage/breakdown');
          if (bResp.ok) {
            var b = await bResp.json();
            App.settings._renderStorageBar(b);
          }
        } catch (e) { /* non-fatal */ }

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

    /**
     * R110: render the stacked-bar storage allocator. Replaces the old
     * dual-slider that confused users about whether `Max Auto-Download
     * Storage` was inside or in addition to `Max Disk`.
     */
    _renderStorageBar: function(b) {
      var bar = document.getElementById('storage-stacked-bar');
      if (!bar || !b) return;
      bar.style.display = '';
      var total = Math.max(1, b.total_mb || 0);
      var used = Math.min(b.used_mb || 0, total);
      var freeRaw = b.free_mb || 0;
      // The auto-manage budget is "head-room reserved for downloads".
      // Visually it sits between used and free — capped so we never
      // exceed the total. Fall back to free if the budget overlaps.
      var budgetSlice = Math.max(0, Math.min(b.auto_target_mb || 0, freeRaw));
      var freeSlice = Math.max(0, freeRaw - budgetSlice);

      var pct = function(mb) { return ((mb / total) * 100).toFixed(2) + '%'; };
      var usedEl = document.getElementById('storage-bar-used');
      var budgetEl = document.getElementById('storage-bar-budget');
      var freeEl = document.getElementById('storage-bar-free');
      if (usedEl) usedEl.style.width = pct(used);
      if (budgetEl) budgetEl.style.width = pct(budgetSlice);
      if (freeEl) freeEl.style.width = pct(freeSlice);

      var fmtGb = function(mb) {
        if (mb >= 1000) return (mb / 1000).toFixed(1) + ' GB';
        return mb + ' MB';
      };
      var setText = function(id, mb) {
        var el = document.getElementById(id);
        if (el) el.textContent = fmtGb(mb);
      };
      setText('storage-legend-used-val', used);
      setText('storage-legend-budget-val', budgetSlice);
      setText('storage-legend-free-val', freeSlice);
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
          if (App.autoManageStatus) App.autoManageStatus.setEnabled(autoManageOn);
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
