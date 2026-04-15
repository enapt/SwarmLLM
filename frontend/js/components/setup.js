'use strict';

// ============================================================================
// SwarmLLM — Setup Wizard Component
// First-run configuration wizard (extracted from settings.js)
// ============================================================================

(function() {
  var U = App.utils;

  // ========================================================================
  // Setup Wizard
  // ========================================================================
  App.setup = {
    currentStep: 1,
    totalSteps: 3,
    hwData: null,
    _savedProvider: null,
    _joinedPeer: false,

    init: function() {
      if (localStorage.getItem(App.SETUP_DONE_KEY) === 'true') return;
      document.getElementById('setup-modal').classList.remove('hidden');
      App.setup.detectHardware();

      document.getElementById('contribution-slider').addEventListener('input', function() {
        var levels = [I18n.t('setup.contrib_minimal'), I18n.t('setup.contrib_moderate'), I18n.t('setup.contrib_maximum')];
        var descs = [
          I18n.t('setup.contrib_minimal_desc'),
          I18n.t('setup.contrib_moderate_desc'),
          I18n.t('setup.contrib_maximum_desc'),
        ];
        var val = parseInt(this.value, 10);
        document.getElementById('contribution-label').textContent = levels[val];
        document.getElementById('contribution-desc').textContent = descs[val];
      });

      // Invite code join button
      var joinBtn = document.getElementById('setup-invite-join');
      if (joinBtn) {
        joinBtn.addEventListener('click', function() { App.setup.joinInvite(); });
      }
    },

    detectHardware: async function() {
      try {
        var result = await App.data.loadStats();
        var data = (result && result.stats) ? result.stats : {};
        App.setup.hwData = data.hardware || {};
        var gpuName = App.setup.hwData.gpu_name || I18n.t('hw.mode_cpu_only');
        var vramMb = App.setup.hwData.gpu_vram_mb || 0;
        document.getElementById('hw-gpu').textContent = gpuName;
        document.getElementById('hw-vram').textContent = vramMb ? U.formatMB(vramMb) + ' ' + I18n.t('hw.vram') : '';
        document.getElementById('hw-ram').textContent = U.formatMB(App.setup.hwData.total_ram_mb || 0) + ' ' + I18n.t('hw.ram');
        document.getElementById('hw-disk').textContent = U.formatMB(App.setup.hwData.available_disk_mb || 0) + ' ' + I18n.t('dashboard.disk_label');
        // Hardware-aware model recommendation
        var rec = document.getElementById('hw-recommendation');
        if (rec) {
          if (vramMb >= 8000) {
            rec.textContent = I18n.t('setup.hw_gpu_7b');
          } else if (vramMb >= 4000) {
            rec.textContent = I18n.t('setup.hw_gpu_small');
          } else if (vramMb > 0) {
            rec.textContent = I18n.t('setup.hw_limited_vram');
          } else {
            rec.textContent = I18n.t('setup.hw_no_gpu');
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
      var levels = [I18n.t('setup.contrib_minimal'), I18n.t('setup.contrib_moderate'), I18n.t('setup.contrib_maximum')];
      var val = parseInt(document.getElementById('contribution-slider').value, 10);
      document.getElementById('summary-contribution').textContent = levels[val];
      var gpuName = App.setup.hwData && App.setup.hwData.gpu_name ? App.setup.hwData.gpu_name : I18n.t('hw.mode_cpu_only');
      document.getElementById('summary-gpu').textContent = gpuName;
      var autoManage = document.getElementById('setup-auto-manage').checked;
      document.getElementById('summary-auto-manage').textContent = autoManage ? I18n.t('setup.summary_enabled') : I18n.t('settings.claude_subscription_disabled');

      // Only show invite/provider rows if configured
      var inviteRow = document.getElementById('summary-invite-row');
      if (App.setup._joinedPeer) { inviteRow.classList.remove('hidden'); document.getElementById('summary-invite').textContent = I18n.t('connection.connected'); }
      var provRow = document.getElementById('summary-provider-row');
      if (App.setup._savedProvider) {
        provRow.classList.remove('hidden');
        document.getElementById('summary-provider').textContent = PROVIDER_NAMES[App.setup._savedProvider] || App.setup._savedProvider;
      }

      // Dynamic next steps
      var steps = [];
      if (autoManage) steps.push(I18n.t('setup.next_auto_manage'));
      if (App.setup._joinedPeer) steps.push(I18n.t('setup.next_joined_peer'));
      if (App.setup._savedProvider) steps.push(I18n.t('setup.next_provider'));
      if (!App.setup._savedProvider && !autoManage) steps.push(I18n.t('setup.next_manual'));
      steps.push(I18n.t('setup.next_lan'));
      document.getElementById('summary-next-list').innerHTML = steps.map(function(s) { return '<p style="margin:4px 0">\u2022 ' + s + '</p>'; }).join('');
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
      document.getElementById('setup-modal').classList.add('hidden');
      App.ui.showBanner('success', I18n.t('setup.complete'));
    },

    finish: function() {
      localStorage.setItem(App.SETUP_DONE_KEY, 'true');
      document.getElementById('setup-modal').classList.add('hidden');
      App.ui.showBanner('info', I18n.t('setup.skipped'));
    }
  };
})();
