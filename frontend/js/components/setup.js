'use strict';

// ============================================================================
// SwarmLLM — Setup Wizard Component
// First-run configuration wizard
// ============================================================================

(function() {
  var U = App.utils;

  // Order matters — first 6 are visible above the fold on most screens.
  var PROVIDER_ORDER = [
    'openai', 'anthropic', 'deepseek', 'groq', 'mistral', 'together',
    'nvidia_nim', 'cerebras', 'sambanova', 'fireworks', 'deepinfra',
  ];

  App.setup = {
    currentStep: 1,
    totalSteps: 3,
    hwData: null,
    _savedProvider: null,
    _selectedProvider: null,
    _joinedPeer: false,

    init: function() {
      // First-run gate: only auto-show if neither completed nor skipped.
      var done = localStorage.getItem(App.SETUP_DONE_KEY) === 'true';
      App.setup._renderFinishChip();
      if (done) return;
      // If skipped, don't auto-show — the chip handles re-entry.
      if (localStorage.getItem(App.SETUP_SKIPPED_KEY) === 'true') return;
      document.getElementById('setup-modal').classList.remove('hidden');
      App.setup.detectHardware();

      App.setup._wireSegmented();
      App.setup._wireProviderTiles();

      // Auto-open advanced if user pasted into invite-code via URL or has it focused
      var joinBtn = document.getElementById('setup-invite-join');
      if (joinBtn) joinBtn.addEventListener('click', function() { App.setup.joinInvite(); });

      // Step indicator clicks: jump to earlier step (existing behavior in init.js, idempotent here)
      // Summary row click-to-jump
      document.querySelectorAll('.setup-summary-row[data-jump-step]').forEach(function(row) {
        row.addEventListener('click', function() {
          var target = parseInt(row.getAttribute('data-jump-step'), 10);
          if (target && target < App.setup.currentStep) {
            App.setup.currentStep = target;
            App.setup.updateUI();
          }
        });
      });
    },

    // Wire the contribution segmented control to the hidden <select>.
    _wireSegmented: function() {
      var seg = document.querySelector('.segmented[data-bound-select="setup-contribution"]');
      var sel = document.getElementById('setup-contribution');
      if (!seg || !sel) return;
      var descEl = document.getElementById('contribution-desc');
      var sync = function() {
        var v = sel.value;
        seg.querySelectorAll('.segmented-btn').forEach(function(b) {
          var on = b.getAttribute('data-value') === v;
          b.classList.toggle('active', on);
          b.setAttribute('aria-checked', on ? 'true' : 'false');
        });
        if (descEl) {
          var keys = { minimal: 'setup.contrib_minimal_desc', moderate: 'setup.contrib_moderate_desc', maximum: 'setup.contrib_maximum_desc' };
          var key = keys[v] || keys.moderate;
          descEl.textContent = I18n.t(key);
          descEl.setAttribute('data-i18n', key);
        }
      };
      seg.querySelectorAll('.segmented-btn').forEach(function(btn) {
        btn.addEventListener('click', function() {
          sel.value = btn.getAttribute('data-value');
          sel.dispatchEvent(new Event('change'));
          sync();
        });
      });
      sel.addEventListener('change', sync);
      sync();

      // Wire the contribution-mode (Auto/Manual) segmented control.
      var modeSeg = document.querySelector('.segmented[data-bound-select="setup-contribution-mode"]');
      var modeSel = document.getElementById('setup-contribution-mode');
      if (modeSeg && modeSel) {
        var modeSync = function() {
          var v = modeSel.value;
          modeSeg.querySelectorAll('.segmented-btn').forEach(function(b) {
            b.classList.toggle('active', b.getAttribute('data-value') === v);
          });
        };
        modeSeg.querySelectorAll('.segmented-btn').forEach(function(btn) {
          btn.addEventListener('click', function() {
            modeSel.value = btn.getAttribute('data-value');
            modeSel.dispatchEvent(new Event('change'));
            modeSync();
          });
        });
        modeSel.addEventListener('change', modeSync);
        modeSync();
      }
    },

    // Build the cloud-provider tile grid.
    _wireProviderTiles: function() {
      var grid = document.getElementById('setup-provider-tiles');
      if (!grid) return;
      grid.innerHTML = '';
      PROVIDER_ORDER.forEach(function(key) {
        if (!PROVIDER_NAMES[key]) return;
        var tile = document.createElement('button');
        tile.type = 'button';
        tile.className = 'setup-provider-tile';
        tile.setAttribute('role', 'radio');
        tile.setAttribute('aria-checked', 'false');
        tile.dataset.provider = key;
        var iconUrl = providerIconUrl(key);
        var iconHtml = iconUrl
          ? '<img src="' + iconUrl + '" alt="" aria-hidden="true">'
          : '<span style="width:28px;height:28px;display:inline-block"></span>';
        tile.innerHTML = iconHtml + '<span class="setup-provider-tile-label">' + U.escapeHtml(PROVIDER_NAMES[key]) + '</span>';
        tile.addEventListener('click', function() { App.setup._selectProvider(key); });
        grid.appendChild(tile);
      });
    },

    _selectProvider: function(key) {
      App.setup._selectedProvider = key;
      var grid = document.getElementById('setup-provider-tiles');
      if (grid) {
        grid.querySelectorAll('.setup-provider-tile').forEach(function(t) {
          var on = t.dataset.provider === key;
          t.classList.toggle('selected', on);
          t.setAttribute('aria-checked', on ? 'true' : 'false');
        });
      }
      var input = document.getElementById('setup-provider-input');
      if (input) input.classList.remove('hidden');
      var nameEl = document.getElementById('setup-provider-selected-name');
      if (nameEl) nameEl.textContent = PROVIDER_NAMES[key] || key;
      var signup = document.getElementById('setup-provider-signup');
      if (signup) signup.href = (typeof PROVIDER_SIGNUP_URLS !== 'undefined' && PROVIDER_SIGNUP_URLS[key]) || '#';
      var status = document.getElementById('setup-provider-status');
      if (status) { status.textContent = ''; status.className = 'setup-provider-status'; }
      var keyInput = document.getElementById('setup-provider-key');
      if (keyInput) { keyInput.value = ''; keyInput.focus(); }
    },

    detectHardware: async function() {
      try {
        var result = await App.data.loadStats();
        var data = (result && result.stats) ? result.stats : {};
        App.setup.hwData = data.hardware || {};
        var gpuName = App.setup.hwData.gpu_name;
        var vramMb = App.setup.hwData.gpu_vram_mb || 0;
        var ramMb = App.setup.hwData.total_ram_mb || 0;
        var diskMb = App.setup.hwData.available_disk_mb || 0;

        var gpuEl = document.getElementById('hw-gpu');
        var vramEl = document.getElementById('hw-vram');
        var ramEl = document.getElementById('hw-ram');
        var diskEl = document.getElementById('hw-disk');
        var rec = document.getElementById('hw-recommendation');

        if (gpuName && vramMb > 0) {
          gpuEl.textContent = gpuName;
          vramEl.textContent = U.formatMB(vramMb) + ' ' + I18n.t('hw.vram');
        } else {
          gpuEl.textContent = I18n.t('hw.mode_cpu_only');
          vramEl.textContent = I18n.t('setup.hw_no_gpu_short');
        }
        ramEl.textContent = U.formatMB(ramMb) + ' ' + I18n.t('hw.ram');
        diskEl.textContent = U.formatMB(diskMb) + ' ' + I18n.t('dashboard.disk_label');

        if (rec) {
          rec.className = 'setup-hw-card-badge';
          if (vramMb >= 8000) {
            rec.textContent = I18n.t('setup.hw_gpu_7b');
          } else if (vramMb >= 4000) {
            rec.textContent = I18n.t('setup.hw_gpu_small');
          } else if (vramMb > 0) {
            rec.textContent = I18n.t('setup.hw_limited_vram');
            rec.classList.add('warn');
          } else {
            rec.textContent = I18n.t('setup.hw_capability_cpu');
            rec.classList.add('cpu');
          }
        }
      } catch (e) {
        document.getElementById('hw-gpu').textContent = I18n.t('setup.hw_detection_failed');
        App.setup.hwData = {};
      }
      document.getElementById('hw-loading').classList.add('hidden');
      document.getElementById('hw-results').classList.remove('hidden');
    },

    joinInvite: async function() {
      var code = (document.getElementById('setup-invite-code').value || '').trim();
      var status = document.getElementById('setup-invite-status');
      await U.submitCodeForm('/api/admin/join-network', code, status, {
        emptyMsg: I18n.t('setup.paste_code_first'),
        failMsg: I18n.t('setup.failed_connect'),
        errorMsg: I18n.t('setup.connection_error', { error: I18n.t('common.request_failed') }),
        onSuccess: function() {
          App.setup._joinedPeer = true;
          document.getElementById('setup-invite-code').value = '';
        }
      });
    },

    // Save + test API key — adapted from settings.testProvider so the wizard
    // gives the user real validation feedback instead of just a "saved" status.
    saveAndTestProvider: async function() {
      var provider = App.setup._selectedProvider;
      var keyInput = document.getElementById('setup-provider-key');
      var status = document.getElementById('setup-provider-status');
      var saveBtn = document.getElementById('setup-provider-save');
      if (!provider) { status.textContent = I18n.t('init.select_provider'); status.className = 'setup-provider-status error'; return; }
      var key = (keyInput.value || '').trim();
      if (!key) { status.textContent = I18n.t('setup.paste_key_first'); status.className = 'setup-provider-status error'; return; }

      status.textContent = I18n.t('setup.testing_key');
      status.className = 'setup-provider-status testing';
      if (saveBtn) saveBtn.disabled = true;

      try {
        var saveBody = {}; saveBody[provider + '_key'] = key;
        var saveResp = await App.authFetch('/api/admin/providers', {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(saveBody),
        });
        if (!saveResp.ok) {
          status.textContent = I18n.t('setup.failed_save');
          status.className = 'setup-provider-status error';
          return;
        }

        // Test with a 1-token request — same shape settings.testProvider uses.
        var testResp;
        if (provider === 'anthropic') {
          testResp = await App.authFetch('/v1/messages', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ model: 'claude-haiku-4-5-20251001', max_tokens: 1, messages: [{ role: 'user', content: 'hi' }] }),
          });
        } else {
          var model = PROVIDER_TEST_MODELS[provider] || provider + '-test';
          testResp = await App.authFetch('/v1/chat/completions', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ model: model, max_tokens: 1, messages: [{ role: 'user', content: 'hi' }] }),
          });
        }
        if (testResp.ok) {
          status.innerHTML = '<span>✓ ' + U.escapeHtml(I18n.t('setup.key_verified', { name: PROVIDER_NAMES[provider] || provider })) + '</span>';
          status.className = 'setup-provider-status success';
          App.setup._savedProvider = provider;
          var tile = document.querySelector('.setup-provider-tile[data-provider="' + provider + '"]');
          if (tile) tile.classList.add('configured');
          if (keyInput) keyInput.value = '';
        } else {
          var errText = await testResp.text();
          var friendlyErr = errText;
          try { var ej = JSON.parse(errText); friendlyErr = (ej.error && ej.error.message) || errText; } catch(pe) {}
          if (friendlyErr.length > 200) friendlyErr = friendlyErr.substring(0, 200) + '…';
          status.textContent = I18n.t('setup.key_test_failed', { error: friendlyErr });
          status.className = 'setup-provider-status error';
        }
      } catch (e) {
        status.textContent = I18n.t('setup.connection_error', { error: e.message || I18n.t('common.request_failed') });
        status.className = 'setup-provider-status error';
      } finally {
        if (saveBtn) saveBtn.disabled = false;
      }
    },

    // Read a key from the clipboard (single click — no Ctrl+V hunt).
    pasteFromClipboard: async function() {
      var status = document.getElementById('setup-provider-status');
      try {
        if (!navigator.clipboard || !navigator.clipboard.readText) {
          status.textContent = I18n.t('setup.paste_unsupported');
          status.className = 'setup-provider-status error';
          return;
        }
        var text = await navigator.clipboard.readText();
        var input = document.getElementById('setup-provider-key');
        if (input && text) input.value = text.trim();
      } catch (e) {
        status.textContent = I18n.t('setup.paste_denied');
        status.className = 'setup-provider-status error';
      }
    },

    nextStep: function() {
      if (App.setup.currentStep === App.setup.totalSteps) {
        App.setup.submit();
        return;
      }
      App.setup.currentStep++;
      App.setup.updateUI();
      if (App.setup.currentStep === App.setup.totalSteps) App.setup.populateSummary();
    },

    prevStep: function() {
      if (App.setup.currentStep > 1) {
        App.setup.currentStep--;
        App.setup.updateUI();
      }
    },

    updateUI: function() {
      for (var i = 1; i <= App.setup.totalSteps; i++) {
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
      document.getElementById('btn-next').textContent = App.setup.currentStep === App.setup.totalSteps ? I18n.t('setup.start') : I18n.t('setup.continue');
    },

    populateSummary: function() {
      var nick = (document.getElementById('setup-nickname').value || '').trim();
      document.getElementById('summary-nickname').textContent = nick || I18n.t('setup.summary_anonymous');
      var contribSel = document.getElementById('setup-contribution');
      var contribKey = (contribSel && contribSel.value) || 'moderate';
      var contribLabels = { minimal: I18n.t('setup.contrib_minimal'), moderate: I18n.t('setup.contrib_moderate'), maximum: I18n.t('setup.contrib_maximum') };
      document.getElementById('summary-contribution').textContent = contribLabels[contribKey] || contribKey;

      // R115: concrete preview — "At Moderate × N GB disk, you'll host
      // ~5 model parts (~30 GB)". Replaces the abstract "≤ 50% CPU"
      // framing with numbers a non-technical user can act on.
      // Prefers the actual available disk (capped at 200 GB so a 2 TB
      // drive doesn't claim all the space) over a hardcoded default.
      var contribFactor = { minimal: 0.25, moderate: 0.5, maximum: 0.75 }[contribKey] || 0.5;
      var availDiskMb = App.setup.hwData && App.setup.hwData.available_disk_mb;
      var diskBudgetGb = availDiskMb && availDiskMb > 0
        ? Math.min(200, Math.max(10, Math.round(availDiskMb / 1024)))
        : 50;
      var autoBudgetGb = Math.round(diskBudgetGb * contribFactor);
      // Average shard ≈ 4 GB (Q4 7B-class); cap at 12 to keep the
      // number realistic.
      var avgShardGb = 4;
      var estShards = Math.min(12, Math.max(1, Math.round(autoBudgetGb / avgShardGb)));
      var previewEl = document.getElementById('summary-storage-preview');
      if (previewEl) {
        previewEl.textContent = I18n.t('setup.summary_storage_preview', {
          shards: estShards,
          gb: autoBudgetGb,
        });
      }
      var gpuName = App.setup.hwData && App.setup.hwData.gpu_name ? App.setup.hwData.gpu_name : I18n.t('hw.mode_cpu_only');
      document.getElementById('summary-gpu').textContent = gpuName;
      var autoManage = document.getElementById('setup-auto-manage').checked;
      document.getElementById('summary-auto-manage').textContent = autoManage ? I18n.t('setup.summary_enabled') : I18n.t('setup.summary_disabled');

      var inviteRow = document.getElementById('summary-invite-row');
      if (App.setup._joinedPeer) { inviteRow.classList.remove('hidden'); document.getElementById('summary-invite').textContent = I18n.t('connection.connected'); }
      else { inviteRow.classList.add('hidden'); }
      var provRow = document.getElementById('summary-provider-row');
      if (App.setup._savedProvider) {
        provRow.classList.remove('hidden');
        document.getElementById('summary-provider').textContent = PROVIDER_NAMES[App.setup._savedProvider] || App.setup._savedProvider;
      } else {
        provRow.classList.add('hidden');
      }

      var steps = [];
      if (autoManage) steps.push(I18n.t('setup.next_auto_manage'));
      if (App.setup._joinedPeer) steps.push(I18n.t('setup.next_joined_peer'));
      if (App.setup._savedProvider) steps.push(I18n.t('setup.next_provider'));
      if (!App.setup._savedProvider && !autoManage) steps.push(I18n.t('setup.next_manual'));
      steps.push(I18n.t('setup.next_lan'));
      var listEl = document.getElementById('summary-next-list');
      listEl.innerHTML = '';
      steps.forEach(function(s) {
        var li = document.createElement('li');
        li.textContent = s;
        listEl.appendChild(li);
      });
    },

    submit: async function() {
      var contribSel = document.getElementById('setup-contribution');
      var level = (contribSel && contribSel.value) || 'moderate';
      var modeSel = document.getElementById('setup-contribution-mode');
      var contributionAuto = !modeSel || modeSel.value !== 'manual';
      var autoManage = document.getElementById('setup-auto-manage').checked;
      try {
        var resp = await App.authFetch('/api/admin/config', {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            contribution: level,
            contribution_auto: contributionAuto,
            auto_manage_shards: autoManage,
          }),
        });
        if (!resp.ok) {
          App.ui.showBanner('error', I18n.t('setup.failed_save'));
          return;
        }
      } catch (e) {
        App.ui.showBanner('error', I18n.t('setup.failed_save_error', { error: e.message || I18n.t('common.request_failed') }));
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
      // Clear the skipped flag so the chip disappears for sure.
      localStorage.removeItem(App.SETUP_SKIPPED_KEY);
      localStorage.removeItem(App.SETUP_CHIP_DISMISSED_KEY);
      document.getElementById('setup-modal').classList.add('hidden');
      App.setup._renderFinishChip();
      App.ui.showBanner('success', I18n.t('setup.complete'));
      if (App.welcome) App.welcome.maybeShow();
    },

    // Skip without marking setup complete — surfaces the "Finish setup" chip in the dashboard.
    finish: function() {
      localStorage.setItem(App.SETUP_SKIPPED_KEY, 'true');
      document.getElementById('setup-modal').classList.add('hidden');
      App.setup._renderFinishChip();
      App.ui.showBanner('info', I18n.t('setup.skipped'));
      if (App.welcome) App.welcome.maybeShow();
    },

    // Show / hide the dashboard "Finish setup" chip based on storage state.
    _renderFinishChip: function() {
      var chip = document.getElementById('setup-finish-chip');
      if (!chip) return;
      var done = localStorage.getItem(App.SETUP_DONE_KEY) === 'true';
      var skipped = localStorage.getItem(App.SETUP_SKIPPED_KEY) === 'true';
      var dismissed = localStorage.getItem(App.SETUP_CHIP_DISMISSED_KEY) === 'true';
      var show = !done && skipped && !dismissed;
      chip.classList.toggle('hidden', !show);
    },

    // Re-open the wizard from the chip.
    reopen: function() {
      localStorage.removeItem(App.SETUP_DONE_KEY);
      localStorage.removeItem(App.SETUP_SKIPPED_KEY);
      App.setup.currentStep = 1;
      App.setup.updateUI();
      document.getElementById('setup-modal').classList.remove('hidden');
      App.setup.detectHardware();
      App.setup._renderFinishChip();
    },

    dismissChip: function() {
      localStorage.setItem(App.SETUP_CHIP_DISMISSED_KEY, 'true');
      App.setup._renderFinishChip();
    },
  };
})();
