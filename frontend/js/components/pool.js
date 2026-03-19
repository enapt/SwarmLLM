// ── Device Pool component ──
// Manages the Devices tab: create/join pool, invite codes, member list, leave.

(function () {
  'use strict';

  var S = App.state;

  App.pool = {
    _poolState: null,
    _isOwner: false,
    _myNodeId: null,

    init: function () {
      // Wire up static buttons
      var createBtn = document.getElementById('pool-create-btn');
      var joinBtn = document.getElementById('pool-join-btn');
      var createSubmit = document.getElementById('pool-create-submit');
      var joinSubmit = document.getElementById('pool-join-submit');
      var inviteCodeBtn = document.getElementById('pool-invite-code-btn');
      var leaveBtn = document.getElementById('pool-leave-btn');
      var copyCodeBtn = document.getElementById('pool-copy-code-btn');

      if (createBtn) createBtn.addEventListener('click', function () {
        document.getElementById('pool-create-form').style.display = '';
        document.getElementById('pool-join-form').style.display = 'none';
      });
      if (joinBtn) joinBtn.addEventListener('click', function () {
        document.getElementById('pool-join-form').style.display = '';
        document.getElementById('pool-create-form').style.display = 'none';
        var input = document.getElementById('pool-join-code');
        if (input) input.focus();
      });
      if (createSubmit) createSubmit.addEventListener('click', function () {
        App.pool.createPool();
      });
      if (joinSubmit) joinSubmit.addEventListener('click', function () {
        App.pool.joinPool();
      });
      if (inviteCodeBtn) inviteCodeBtn.addEventListener('click', function () {
        App.pool.generateInviteCode();
      });
      if (leaveBtn) leaveBtn.addEventListener('click', function () {
        App.pool.leavePool();
      });
      if (copyCodeBtn) copyCodeBtn.addEventListener('click', function () {
        App.pool.copyInviteCode();
      });

      // Allow Enter key in join code input
      var joinInput = document.getElementById('pool-join-code');
      if (joinInput) joinInput.addEventListener('keydown', function (e) {
        if (e.key === 'Enter') App.pool.joinPool();
      });
      var createInput = document.getElementById('pool-create-name');
      if (createInput) createInput.addEventListener('keydown', function (e) {
        if (e.key === 'Enter') App.pool.createPool();
      });
    },

    load: async function () {
      try {
        // Get our node ID for owner detection
        if (!this._myNodeId) {
          var statsResp = await App.authFetch('/api/admin/stats');
          if (statsResp.ok) {
            var stats = await statsResp.json();
            this._myNodeId = stats.node_id || null;
          }
        }

        var resp = await App.authFetch('/api/pool/state');
        if (!resp.ok) return;
        var data = await resp.json();
        this._poolState = data;

        if (data.in_pool) {
          this._isOwner = data.pool_id === this._myNodeId;
          this.renderActivePool(data);
        } else {
          this.renderNoPool();
        }
      } catch (e) {
        console.error('Pool load error:', e);
      }
    },

    renderNoPool: function () {
      var noPool = document.getElementById('pool-no-pool');
      var active = document.getElementById('pool-active');
      if (noPool) noPool.style.display = '';
      if (active) active.style.display = 'none';
      // Reset forms
      var createForm = document.getElementById('pool-create-form');
      var joinForm = document.getElementById('pool-join-form');
      if (createForm) createForm.style.display = 'none';
      if (joinForm) joinForm.style.display = 'none';
    },

    renderActivePool: function (data) {
      var noPool = document.getElementById('pool-no-pool');
      var active = document.getElementById('pool-active');
      if (noPool) noPool.style.display = 'none';
      if (active) active.style.display = '';

      // Pool name
      var nameEl = document.getElementById('pool-name');
      if (nameEl) nameEl.textContent = data.name || 'Device Pool';

      // Role label
      var roleEl = document.getElementById('pool-role-label');
      if (roleEl) {
        if (this._isOwner) {
          roleEl.textContent = I18n.t('pool.role_owner') || 'You are the pool owner (master device)';
          roleEl.style.color = 'var(--green)';
        } else {
          roleEl.textContent = I18n.t('pool.role_member') || 'Member — credits forwarded to owner';
          roleEl.style.color = 'var(--cyan)';
        }
      }

      // Show invite code button only for owner
      var inviteBtn = document.getElementById('pool-invite-code-btn');
      if (inviteBtn) inviteBtn.style.display = this._isOwner ? '' : 'none';

      // Stats
      var members = data.members || [];
      var memberCount = document.getElementById('pool-member-count');
      var totalCredits = document.getElementById('pool-total-credits');
      if (memberCount) memberCount.textContent = members.length;
      if (totalCredits) totalCredits.textContent = (data.total_lifetime_credits || 0).toLocaleString();

      // Member list
      this.renderMembers(members);
    },

    renderMembers: function (members) {
      var list = document.getElementById('pool-members-list');
      if (!list) return;
      list.innerHTML = '';

      if (members.length === 0) {
        list.innerHTML = '<div class="empty-state text-muted">' +
          App.utils.escapeHtml(I18n.t('pool.no_members') || 'No members yet. Generate an invite code to add devices.') +
          '</div>';
        return;
      }

      var tmpl = document.getElementById('tmpl-pool-member-row');
      if (!tmpl) return;

      for (var i = 0; i < members.length; i++) {
        var m = members[i];
        var row = tmpl.content.cloneNode(true).firstElementChild;

        var idEl = row.querySelector('.pool-member-id');
        var joinedEl = row.querySelector('.pool-member-joined');
        var creditsEl = row.querySelector('.pool-member-credits');
        var removeBtn = row.querySelector('.pool-member-remove-btn');

        var shortId = m.node_id ? m.node_id.substring(0, 16) + '...' : '?';
        var isSelf = m.node_id === this._myNodeId;

        if (idEl) idEl.textContent = shortId + (isSelf ? ' (you)' : '');
        if (joinedEl) joinedEl.textContent = (I18n.t('pool.joined') || 'Joined') + ': ' +
          (m.joined_at ? m.joined_at.substring(0, 10) : '?');
        if (creditsEl) creditsEl.textContent = (m.credits_contributed || 0).toLocaleString();

        // Show remove button for owner (not self)
        if (removeBtn && this._isOwner && !isSelf) {
          removeBtn.style.display = '';
          removeBtn.setAttribute('data-pool-remove', m.node_id);
          removeBtn.addEventListener('click', function () {
            var nid = this.getAttribute('data-pool-remove');
            App.pool.removeMember(nid);
          });
        }

        // Icon: owner gets a crown, self gets a star
        var iconEl = row.querySelector('.pool-member-icon');
        if (iconEl) {
          if (m.node_id === (App.pool._poolState && App.pool._poolState.pool_id)) {
            iconEl.textContent = '\uD83D\uDC51'; // crown for owner
          } else if (isSelf) {
            iconEl.textContent = '\u2B50'; // star for self
          }
        }

        list.appendChild(row);
      }
    },

    createPool: async function () {
      var input = document.getElementById('pool-create-name');
      var name = input ? input.value.trim() : '';
      if (!name) {
        App.notifications.showToast(I18n.t('pool.name_required') || 'Pool name is required', 'error');
        return;
      }
      try {
        var resp = await App.authFetch('/api/pool/create', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ name: name })
        });
        var data = await resp.json();
        if (data.error) {
          App.notifications.showToast(data.error, 'error');
        } else {
          App.notifications.showToast(I18n.t('pool.created_success') || 'Pool created!', 'success');
          this.load();
        }
      } catch (e) {
        App.notifications.showToast('Failed to create pool: ' + e.message, 'error');
      }
    },

    joinPool: async function () {
      var input = document.getElementById('pool-join-code');
      var code = input ? input.value.trim().toUpperCase() : '';
      if (!code || code.length !== 8) {
        App.notifications.showToast(I18n.t('pool.code_invalid') || 'Enter an 8-character invite code', 'error');
        return;
      }
      try {
        var resp = await App.authFetch('/api/pool/join', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ code: code })
        });
        var data = await resp.json();
        if (data.error) {
          App.notifications.showToast(data.error, 'error');
        } else {
          App.notifications.showToast(
            I18n.t('pool.join_sent') || 'Join request sent! You\'ll be added once the owner\'s node processes it.',
            'success'
          );
          if (input) input.value = '';
          // Poll for acceptance
          setTimeout(function () { App.pool.load(); }, 5000);
          setTimeout(function () { App.pool.load(); }, 15000);
        }
      } catch (e) {
        App.notifications.showToast('Failed to join: ' + e.message, 'error');
      }
    },

    generateInviteCode: async function () {
      try {
        var resp = await App.authFetch('/api/pool/generate-code', { method: 'POST' });
        var data = await resp.json();
        if (data.error) {
          App.notifications.showToast(data.error, 'error');
          return;
        }
        var code = data.code;
        var display = document.getElementById('pool-invite-code-display');
        var codeVal = document.getElementById('pool-invite-code-value');
        if (display) display.style.display = '';
        if (codeVal) codeVal.textContent = code;
        App.pool._lastCode = code;
      } catch (e) {
        App.notifications.showToast('Failed to generate code: ' + e.message, 'error');
      }
    },

    copyInviteCode: function () {
      var code = this._lastCode;
      if (!code) return;
      navigator.clipboard.writeText(code).then(function () {
        App.notifications.showToast(I18n.t('pool.code_copied') || 'Invite code copied!', 'success');
      });
    },

    leavePool: async function () {
      if (!confirm(I18n.t('pool.confirm_leave') || 'Leave the device pool? Credits will no longer be forwarded.')) {
        return;
      }
      try {
        var resp = await App.authFetch('/api/pool/leave', { method: 'POST' });
        var data = await resp.json();
        if (data.error) {
          App.notifications.showToast(data.error, 'error');
        } else {
          App.notifications.showToast(I18n.t('pool.left') || 'Left the pool', 'success');
          this.load();
        }
      } catch (e) {
        App.notifications.showToast('Failed to leave pool: ' + e.message, 'error');
      }
    },

    removeMember: async function (nodeId) {
      if (!confirm(I18n.t('pool.confirm_remove') || 'Remove this device from the pool?')) {
        return;
      }
      try {
        var resp = await App.authFetch('/api/pool/remove', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ node_id: nodeId })
        });
        var data = await resp.json();
        if (data.error) {
          App.notifications.showToast(data.error, 'error');
        } else {
          App.notifications.showToast(I18n.t('pool.member_removed') || 'Device removed', 'success');
          this.load();
        }
      } catch (e) {
        App.notifications.showToast('Failed to remove: ' + e.message, 'error');
      }
    }
  };
})();
