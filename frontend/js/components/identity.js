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
        var resp = await fetch('/api/identity/nickname');
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

      try {
        var resp = await App.authFetch('/api/identity/nickname', {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            nickname: nickname,
            visibility: visEl ? visEl.value : 'nickname',
          }),
        });
        if (!resp.ok) {
          var err = await resp.json().catch(function() { return {}; });
          App.ui.showBanner('error', err.error ? err.error.message : 'Failed to set nickname');
        }
      } catch (e) {
        App.ui.showBanner('error', 'Error saving nickname: ' + e.message);
      }
    },

    loadLeaderboard: async function() {
      var tbody = document.getElementById('leaderboard-body');
      if (!tbody) return;

      try {
        var resp = await fetch('/api/identity/leaderboard?limit=50');
        if (!resp.ok) { tbody.innerHTML = '<tr><td colspan="4" class="text-muted" style="text-align:center">Failed to load</td></tr>'; return; }
        var data = await resp.json();
        var entries = data.leaderboard || [];

        if (entries.length === 0) {
          tbody.innerHTML = '<tr><td colspan="4" class="text-muted" style="text-align:center;padding:24px">No activity yet. Credits are earned by helping others run AI models.</td></tr>';
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
          row.querySelector('.lb-credits').textContent = e.credits || 0;
          var tierEl = row.querySelector('.lb-tier');
          var tierClass = (e.tier || 'silver').toLowerCase().replace(/[^a-z]/g, '');
          tierEl.className = 'tier-badge ' + tierClass;
          tierEl.textContent = e.tier || 'Silver';
          tbody.appendChild(row);
        }
      } catch (e) {
        tbody.innerHTML = '<tr><td colspan="4" class="text-muted" style="text-align:center">Error: ' + U.escapeHtml(e.message) + '</td></tr>';
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
      if (input && input.value) {
        navigator.clipboard.writeText(input.value).then(function() {
          if (btn) { btn.textContent = 'Copied!'; btn.style.color = 'var(--green)'; setTimeout(function() { btn.textContent = 'Copy'; btn.style.color = ''; }, 2000); }
          App.notifications.showToast('Network code copied to clipboard', 'success');
        }).catch(function() {
          App.ui.showBanner('error', 'Failed to copy \u2014 try selecting and copying manually');
        });
      }
    },

    join: async function() {
      var input = document.getElementById('join-code-input');
      var status = document.getElementById('join-status');
      if (!input || !input.value.trim()) return;

      if (status) { status.textContent = 'Connecting...'; status.style.color = 'var(--text-muted)'; }

      try {
        var resp = await App.authFetch('/api/admin/join-network', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ code: input.value.trim() })
        });
        var data = await resp.json();
        if (resp.ok) {
          if (status) { status.textContent = 'Connected! Peer added.'; status.style.color = 'var(--green)'; }
          input.value = '';
          App.notifications.showToast('Peer connected successfully', 'success');
          setTimeout(function() { App.networkCode.load(); }, 2000);
        } else {
          if (status) { status.textContent = data.error || 'Failed to join'; status.style.color = 'var(--red, #ff6464)'; }
        }
      } catch (e) {
        if (status) { status.textContent = 'Network error'; status.style.color = 'var(--red, #ff6464)'; }
      }
    }
  };
})();
