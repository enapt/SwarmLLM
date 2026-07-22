'use strict';

// ============================================================================
// SwarmLLM — Identity & Network Code Component
// Nickname, leaderboard, network invite codes
// ============================================================================

(function() {
  var U = App.utils;

  // --- Identity / Leaderboard ---
  App.identity = {
    loadNickname: async function() {
      try {
        var resp = await App.authFetch('/api/identity/nickname');
        if (!resp.ok) return;
        var data = await resp.json();
        var nickEl = document.getElementById('settings-nickname');
        var visEl = document.getElementById('settings-visibility');
        if (nickEl && data.nickname) nickEl.value = data.nickname;
        if (visEl && data.visibility) visEl.value = data.visibility;
      } catch (e) {}
    },

    saveNickname: async function() {
      var nickEl = document.getElementById('settings-nickname');
      var visEl = document.getElementById('settings-visibility');
      if (!nickEl) return;
      var nickname = nickEl.value.trim();

      if (!nickname) {
        try {
          await App.authFetch('/api/identity/nickname', { method: 'DELETE' });
        } catch (e) {}
        return;
      }

      await U.apiAction(
        '/api/identity/nickname',
        {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            nickname: nickname,
            visibility: visEl ? visEl.value : 'nickname',
          }),
        },
        null,
        { fallback: I18n.t('identity.nickname_failed') }
      );
    },

    loadLeaderboard: async function() {
      var tbody = document.getElementById('leaderboard-body');
      if (!tbody) return;

      try {
        var resp = await App.authFetch('/api/identity/leaderboard?limit=50');
        if (!resp.ok) { tbody.innerHTML = '<tr><td colspan="4" class="text-muted" style="text-align:center">' + U.escapeHtml(I18n.t('leaderboard.load_failed')) + '</td></tr>'; return; }
        var data = await resp.json();
        var entries = data.leaderboard || [];

        if (entries.length === 0) {
          tbody.innerHTML = '<tr><td colspan="4" class="text-muted" style="text-align:center;padding:24px">' + U.escapeHtml(I18n.t('leaderboard.empty')) + '</td></tr>';
          return;
        }

        tbody.innerHTML = '';
        var rowTmpl = document.getElementById('tmpl-leaderboard-row');
        for (var i = 0; i < entries.length; i++) {
          var e = entries[i];
          var row = rowTmpl.content.cloneNode(true).firstElementChild;
          row.querySelector('.lb-rank').textContent = e.rank || i + 1;
          var nameCell = row.querySelector('.lb-name');
          if (e.display_name !== e.node_id) {
            nameCell.textContent = e.display_name;
            var sub = document.createElement('span');
            sub.className = 'text-muted mono';
            sub.style.fontSize = '0.75rem';
            sub.textContent = ' ' + e.node_id;
            nameCell.appendChild(sub);
          } else {
            var mono = document.createElement('span');
            mono.className = 'mono';
            mono.textContent = e.node_id;
            nameCell.appendChild(mono);
          }
          // `credits: null` means this peer's balance gossip hasn't arrived.
          // Show an em dash — NOT 0, which would read as a real balance.
          var creditsEl = row.querySelector('.lb-credits');
          var tierEl = row.querySelector('.lb-tier');
          if (e.credits === null || e.credits === undefined) {
            creditsEl.textContent = '—';
            creditsEl.title = I18n.t('leaderboard.balance_unknown');
            tierEl.className = 'tier-badge';
            tierEl.textContent = '—';
            tierEl.title = I18n.t('leaderboard.balance_unknown');
          } else {
            creditsEl.textContent = e.credits;
            var tierClass = (e.tier || 'silver').toLowerCase().replace(/[^a-z]/g, '');
            tierEl.className = 'tier-badge ' + tierClass;
            tierEl.textContent = e.tier || I18n.t('leaderboard.tier_default');
          }
          tbody.appendChild(row);
        }
      } catch (e) {
        tbody.innerHTML = '<tr><td colspan="4" class="text-muted" style="text-align:center">' + U.escapeHtml(I18n.t('leaderboard.load_error', { error: e.message })) + '</td></tr>';
      }
    }
  };

  // --- Network Code ---
  App.networkCode = {
    load: async function() {
      try {
        var resp = await App.authFetch('/api/admin/network-code');
        var data = await resp.json();
        var codeInput = document.getElementById('my-network-code');
        if (codeInput && data.code) codeInput.value = data.code;
      } catch (e) {}
    },

    copy: function() {
      var input = document.getElementById('my-network-code');
      var btn = document.getElementById('btn-copy-network-code');
      if (!input || !input.value) return;
      U.copyToClipboard(input.value, {
        btn: btn,
        successLabel: I18n.t('actions.copied'),
        resetLabel: I18n.t('actions.copy'),
        onSuccess: function() {
          App.notifications.showToast(I18n.t('identity.code_copied'), 'success');
        },
        onFailure: function() {
          App.ui.showBanner('error', I18n.t('identity.copy_failed'));
        },
      });
    },

    join: async function() {
      var input = document.getElementById('join-code-input');
      var status = document.getElementById('join-status');
      var code = input ? input.value.trim() : '';
      await U.submitCodeForm('/api/admin/join-network', code, status, {
        onSuccess: function() {
          if (input) input.value = '';
          App.notifications.showToast(I18n.t('identity.peer_connected'), 'success');
          setTimeout(function() { App.networkCode.load(); }, 2000);
        }
      });
    }
  };
})();
