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

    // Start the API-key bootstrap if it hasn't started, and return the promise.
    //
    // Callers used to test `if (App.settings._apiKeyPromise)` and skip the wait
    // when it was still null — i.e. they SAMPLED whether someone else had begun
    // the bootstrap rather than ensuring it had. Anything running before the
    // one assignment therefore fetched with no Authorization header and got a
    // 401: six dashboard panels did exactly that on every single page load, on
    // loopback too, and never showed an error because `_restFetch` in
    // swarm-tab.js discards failures. Awaiting this instead makes the wrong
    // call unrepresentable.
    //
    // Memoized because `loadApiKey` consumes the single-use bootstrap nonce and
    // blanks the meta tag — a second concurrent call would find no nonce, take
    // the 401 path, and misreport the origin as untrusted.
    // `_apiKeyStarted` flips BEFORE loadApiKey is invoked, because an async
    // function runs synchronously up to its first await — so anything it calls
    // on the way there re-enters this function while `_apiKeyPromise` is still
    // null, and a memo assigned from the return value is too late to stop it.
    _apiKeyStarted: false,
    ensureApiKey: function() {
      if (App.settings._apiKeyStarted) {
        return App.settings._apiKeyPromise || Promise.resolve();
      }
      if (typeof App.settings.loadApiKey !== 'function') return Promise.resolve();
      App.settings._apiKeyStarted = true;
      App.settings._apiKeyPromise = App.settings.loadApiKey();
      return App.settings._apiKeyPromise;
    },

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
    _applyContributionMode: function(modeAuto) {
      // Relabel the contribution segmented control + hint based on mode.
      // Auto: it's an upper cap; auto-manage scales below it. Manual: it's
      // a pinned level; auto-manage never goes above OR below.
      var label = document.getElementById('settings-contribution-label');
      if (label) {
        label.textContent = I18n.t(modeAuto ? 'settings.contribution_label_cap' : 'settings.contribution_label');
      }
      var hint = document.getElementById('settings-contribution-mode-hint');
      if (hint) {
        hint.textContent = I18n.t(modeAuto ? 'settings.contribution_mode_hint_auto' : 'settings.contribution_mode_hint_manual');
      }
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
          document.getElementById('settings-storage-info').classList.toggle('hidden', !isOn);
          if (isOn) App.settings.loadStorageInfo();
        });
      }
      var modeSelect = document.getElementById('settings-contribution-mode');
      if (modeSelect) {
        modeSelect.addEventListener('change', function() {
          App.settings._applyContributionMode(this.value === 'auto');
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
        // contribution_auto defaults to true — Auto is the recommended mode
        // because an idle node holds redundant shards at swarm scale.
        var modeAuto = data.contribution_auto !== false;
        document.getElementById('settings-contribution-mode').value = modeAuto ? 'auto' : 'manual';
        App.settings._applyContributionMode(modeAuto);
        document.getElementById('settings-max-requests').value = data.max_concurrent_requests || 10;
        document.getElementById('settings-bandwidth').value = data.max_bandwidth_mbps || 0;
        document.getElementById('settings-disk').value = data.max_disk_mb || 50000;
        var trustLanEl = document.getElementById('settings-trust-lan');
        if (trustLanEl) trustLanEl.checked = !!data.dashboard_trust_lan;
        var updModeEl = document.getElementById('settings-update-mode');
        if (updModeEl && data.update_mode) updModeEl.value = data.update_mode;
        var autoManage = data.auto_manage_shards ? 'on' : 'off';
        document.getElementById('settings-auto-shards').value = autoManage;
        var isOn = autoManage === 'on';
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
      App.settings.ensureApiKey();
      App.settings.loadProviders();
    },

    loadApiKey: async function() {
      // Deliberately NOT gated on the settings field existing: obtaining the
      // key is what unblocks every admin write on the page, so it must run
      // even if the settings panel is absent from the DOM.
      var keyEl = document.getElementById('settings-api-key');
      var setField = function(v) { if (keyEl) keyEl.value = v; };
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
        // Plain `fetch`, NOT `App.authFetch`. This request is by definition
        // unauthenticated — it is how the page obtains the credential that
        // authFetch attaches — so routing it through authFetch is both
        // pointless and circular: authFetch calls ensureApiKey, which calls
        // this function, whose memo is not assigned until it returns. That
        // recursed into a request storm the moment authFetch started ensuring
        // rather than sampling.
        var resp = await fetch('/api/admin/api-key', { headers: headers });
        if (resp.ok) {
          var data = await resp.json();
          var key = data.api_key || '';
          App.settings._apiKeyFull = key;
          App.settings._apiKeyDenied = false;
          setField(key ? key.substring(0, 4) + '****' + key.substring(key.length - 4) : I18n.t('settings.no_api_key'));
        } else if (resp.status === 401 && !App.utils.isTrustedOrigin()) {
          // Expected refusal, not a fault: the daemon does not hand keys to
          // this network. A key the user pasted on a previous visit is the
          // supported way through, so try that before declaring failure.
          var saved = App.settings._loadManualKey();
          if (saved) {
            App.settings._apiKeyFull = saved;
            App.settings._apiKeyDenied = false;
            setField(saved.substring(0, 4) + '****' + saved.substring(saved.length - 4));
          } else {
            App.settings._apiKeyDenied = true;
            setField(I18n.t('settings.key_remote'));
          }
        } else {
          App.settings._apiKeyDenied = false;
          setField(I18n.t('settings.key_unavailable'));
        }
      } catch (e) {
        setField(I18n.t('settings.key_error'));
      }
      App.settings._notifyIfUntrusted();
    },

    // Per-origin so one browser used against several nodes never sends node A's
    // key to node B — the keys are unrelated and a mismatch reads as a 401.
    _manualKeyName: function() {
      return App.MANUAL_KEY_KEY + ':' + location.host;
    },

    _loadManualKey: function() {
      try {
        return localStorage.getItem(App.settings._manualKeyName()) || '';
      } catch (e) {
        return '';
      }
    },

    // Verify before storing. A key that doesn't work must fail here, while the
    // user is looking at the box they typed it into — not silently later as a
    // generic failure on some unrelated panel.
    saveManualKey: async function(key) {
      key = (key || '').trim();
      if (!key) return false;
      var resp;
      try {
        resp = await fetch('/api/admin/stats', { headers: { 'Authorization': 'Bearer ' + key } });
      } catch (e) {
        return false;
      }
      if (!resp.ok) return false;
      try {
        localStorage.setItem(App.settings._manualKeyName(), key);
      } catch (e) {
        // Private browsing with storage disabled — the key still works for
        // this page load, it just won't survive a reload.
      }
      App.settings._apiKeyFull = key;
      App.settings._apiKeyDenied = false;
      return true;
    },

    forgetManualKey: function() {
      try {
        localStorage.removeItem(App.settings._manualKeyName());
      } catch (e) { /* nothing to remove */ }
    },

    // Persistent, dismissible explanation, with the way out attached.
    //
    // Persistent because the user has to go and DO something (fetch a key off
    // the machine), which is more than a toast's few seconds allow; at the top
    // of the page because the setup wizard is a modal overlay and this has to
    // be legible over it.
    _notifyIfUntrusted: function() {
      if (!App.settings._apiKeyDenied) return;
      if (document.getElementById('remote-dashboard-banner')) return;
      var U = App.utils;
      var banner = document.createElement('div');
      banner.id = 'remote-dashboard-banner';
      // `top-banner`: see `measureTopBanners`. This one wraps to 60vh, so
      // it covers far more of the page than the update banner does.
      banner.className = 'remote-dashboard-banner top-banner';

      var text = document.createElement('span');
      // Name the address the DAEMON saw. Behind a subnet router or a container
      // publish that is not the address in the user's address bar, and it is
      // the one they would have to allow. Fall back to the address bar only if
      // the page wasn't rendered by the daemon (dev server, raw file open).
      text.textContent = I18n.t('errors.untrusted_dashboard', {
        addr: U.clientAddr() || location.hostname,
      });
      banner.appendChild(text);

      var form = document.createElement('form');
      form.className = 'remote-dashboard-banner-form';
      var input = document.createElement('input');
      input.type = 'password';
      input.autocomplete = 'off';
      input.spellcheck = false;
      input.placeholder = I18n.t('settings.paste_key_placeholder');
      input.setAttribute('aria-label', I18n.t('settings.paste_key_placeholder'));
      var submit = document.createElement('button');
      submit.type = 'submit';
      submit.className = 'btn btn-sm';
      submit.textContent = I18n.t('actions.unlock');
      form.appendChild(input);
      form.appendChild(submit);
      form.addEventListener('submit', async function(ev) {
        ev.preventDefault();
        submit.disabled = true;
        var ok = await App.settings.saveManualKey(input.value);
        submit.disabled = false;
        if (ok) {
          banner.remove();
          // Everything on the page loaded without a key, so re-fetch rather
          // than leaving a dashboard full of empty panels behind the banner.
          location.reload();
        } else {
          input.value = '';
          input.placeholder = I18n.t('settings.paste_key_rejected');
          App.notifications.showToast(I18n.t('settings.paste_key_rejected'), 'error');
        }
      });
      banner.appendChild(form);

      var help = document.createElement('span');
      help.className = 'remote-dashboard-banner-help';
      help.textContent = I18n.t('errors.untrusted_dashboard_where');
      banner.appendChild(help);

      var close = document.createElement('button');
      close.className = 'remote-dashboard-banner-close';
      close.setAttribute('aria-label', I18n.t('actions.dismiss'));
      close.textContent = '✕';
      close.addEventListener('click', function() { banner.remove(); });
      banner.appendChild(close);
      document.body.appendChild(banner);
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

        // The auto-manage limit is the EFFECTIVE figure the daemon decides
        // by (`auto_target_mb` from the breakdown endpoint), not the raw
        // config field. Showing "50 GB" from the config while the daemon
        // was working to a quarter of that is how a tester came to hunt a
        // phantom reservation (gotcha #448).
        var maxEl = document.getElementById('settings-storage-max');
        if (maxEl) maxEl.textContent = '--';
        try {
          var bResp = await App.authFetch('/api/admin/storage/breakdown');
          if (bResp.ok) {
            var b = await bResp.json();
            if (maxEl) maxEl.textContent = U.formatMB(b.auto_target_mb || 0);
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
      // `auto_target_mb` is the CAP on what auto-manage may hold in total,
      // not head-room on top of what is used. The middle slice is the room
      // left under that cap, so a node that is over it shows 0 — the old
      // drawing put the whole cap AFTER "used", and a node that could not
      // download anything displayed a healthy budget slice (gotcha #448).
      var cap = Math.min(b.auto_target_mb || 0, total);
      var budgetSlice = Math.max(0, Math.min(cap - used, freeRaw));
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
        App.networkStatus.load();
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
      var result = await saveAndVerifyProviderKey(name, key);
      if (result.ok) {
        badge.textContent = I18n.t('settings.badge_active');
        badge.className = 'badge provider-badge-active';
        App.ui.showBanner('success', I18n.t('settings.key_verified', { name: name }));
        var testCard = badge.closest('.provider-card');
        if (testCard) testCard.classList.add('provider-active');
        input.value = '';
        App.models.load();
        App.networkStatus.load();
      } else if (result.stage === 'network') {
        badge.textContent = I18n.t('settings.badge_error');
        badge.className = 'badge badge-error';
        App.ui.showBanner('error', I18n.t('settings.key_test_failed', { name: name, error: result.message }));
      } else {
        badge.textContent = I18n.t('settings.badge_failed');
        badge.className = 'badge badge-error';
        App.ui.showBanner('error', I18n.t('settings.key_test_failed', { name: name, error: result.message }));
        input.value = '';
      }
    },

    save: async function() {
      var saveBtn = document.getElementById('btn-save-settings');
      if (saveBtn) { saveBtn.disabled = true; saveBtn.textContent = I18n.t('actions.saving'); }

      var autoManageOn = document.getElementById('settings-auto-shards').value === 'on';
      var modeAuto = document.getElementById('settings-contribution-mode').value === 'auto';
      var config = {
        contribution: document.getElementById('settings-contribution').value,
        contribution_auto: modeAuto,
        max_concurrent_requests: parseInt(document.getElementById('settings-max-requests').value, 10),
        max_bandwidth_mbps: parseInt(document.getElementById('settings-bandwidth').value, 10),
        max_disk_mb: parseInt(document.getElementById('settings-disk').value, 10),
        auto_manage_shards: autoManageOn,
        dashboard_trust_lan: !!(document.getElementById('settings-trust-lan') || {}).checked,
        update_mode: (document.getElementById('settings-update-mode') || {}).value || undefined,
        // R110 removed the standalone max-storage slider — the auto-manage budget
        // is derived from Max Disk + contribution mode, so this field is no longer
        // sent (omitting it leaves the derived value untouched server-side). The
        // dead `settings-auto-manage-storage` read here was throwing a TypeError
        // whenever auto-manage was on, silently killing the entire settings save.
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
