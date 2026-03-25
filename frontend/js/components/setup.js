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
        var resp = await App.authFetch('/api/admin/stats');
        var data = await resp.json();
        App.setup.hwData = data.hardware || {};
        var gpuName = App.setup.hwData.gpu_name || 'No GPU (CPU mode)';
        var vramMb = App.setup.hwData.gpu_vram_mb || 0;
        document.getElementById('hw-gpu').textContent = gpuName;
        document.getElementById('hw-vram').textContent = vramMb ? U.formatMB(vramMb) + ' VRAM' : '';
        document.getElementById('hw-ram').textContent = U.formatMB(App.setup.hwData.total_ram_mb || 0) + ' RAM';
        document.getElementById('hw-disk').textContent = U.formatMB(App.setup.hwData.available_disk_mb || 0) + ' disk';
        // Hardware-aware model recommendation
        var rec = document.getElementById('hw-recommendation');
        if (rec) {
          if (vramMb >= 8000) {
            rec.textContent = 'Your GPU can run 7B models locally (Qwen 7B, Phi-3.5, Gemma 2B)';
          } else if (vramMb >= 4000) {
            rec.textContent = 'Your GPU can run smaller models locally (TinyLlama, Gemma 2B)';
          } else if (vramMb > 0) {
            rec.textContent = 'Limited VRAM — best with CPU inference or cloud providers';
          } else {
            rec.textContent = 'No GPU detected — CPU inference works, or add a cloud provider for speed';
          }
        }
      } catch (e) {
        document.getElementById('hw-gpu').textContent = 'Detection failed';
        App.setup.hwData = {};
      }
      document.getElementById('hw-loading').classList.add('hidden');
      document.getElementById('hw-results').classList.remove('hidden');
    },

    joinInvite: async function() {
      var code = (document.getElementById('setup-invite-code').value || '').trim();
      var status = document.getElementById('setup-invite-status');
      if (!code) { status.textContent = 'Paste an invite code first'; status.style.color = 'var(--text-muted)'; return; }
      status.textContent = 'Connecting...'; status.style.color = 'var(--text-muted)';
      try {
        var resp = await App.authFetch('/api/admin/join-network', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ code: code }),
        });
        var result = await resp.json();
        if (resp.ok) {
          status.textContent = 'Connected! Peer added.';
          status.style.color = 'var(--green)';
          App.setup._joinedPeer = true;
          document.getElementById('setup-invite-code').value = '';
        } else {
          status.textContent = result.error ? result.error.message : 'Failed to connect';
          status.style.color = 'var(--red)';
        }
      } catch (e) {
        status.textContent = 'Connection error: ' + (e.message || 'network error');
        status.style.color = 'var(--red)';
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
      var levels = ['minimal', 'moderate', 'maximum'];
      var val = parseInt(document.getElementById('contribution-slider').value, 10);
      document.getElementById('summary-contribution').textContent = U.capitalize(levels[val]);
      var gpuName = App.setup.hwData && App.setup.hwData.gpu_name ? App.setup.hwData.gpu_name : 'CPU only';
      document.getElementById('summary-gpu').textContent = gpuName;
      var autoManage = document.getElementById('setup-auto-manage').checked;
      document.getElementById('summary-auto-manage').textContent = autoManage ? I18n.t('setup.summary_enabled') : I18n.t('setup.summary_disabled');

      // Only show invite/provider rows if configured
      var inviteRow = document.getElementById('summary-invite-row');
      if (App.setup._joinedPeer) { inviteRow.classList.remove('hidden'); document.getElementById('summary-invite').textContent = I18n.t('setup.summary_connected'); }
      var provRow = document.getElementById('summary-provider-row');
      var provNames = {openai:'OpenAI',deepseek:'DeepSeek',groq:'Groq',nvidia_nim:'NVIDIA NIM',cerebras:'Cerebras',sambanova:'SambaNova',anthropic:'Anthropic',mistral:'Mistral',fireworks:'Fireworks',together:'Together',deepinfra:'DeepInfra'};
      if (App.setup._savedProvider) {
        provRow.classList.remove('hidden');
        document.getElementById('summary-provider').textContent = provNames[App.setup._savedProvider] || App.setup._savedProvider;
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
