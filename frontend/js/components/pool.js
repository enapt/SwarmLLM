// ── Device Pool component ──
// Manages the "My Devices" tab: create/join pool, invite codes, member list,
// device nicknames, online status, per-device stats, credit split, leave.

(function () {
  'use strict';

  var S = App.state;
  var U = App.utils;

  function _hasError(data) {
    if (data && data.error) {
      var msg = U.extractErrorMessage(data, I18n.t('pool.failed_generic', { error: '' }));
      App.notifications.showToast(I18n.t('pool.failed_generic', { error: msg }), 'error');
      return true;
    }
    return false;
  }

  App.pool = {
    _poolState: null,
    _isOwner: false,
    _myNodeId: null,
    _lastCode: null,

    init: function () {
      var self = this;
      // Wire up static buttons
      var ids = {
        'pool-create-btn': function () {
          document.getElementById('pool-create-form').style.display = '';
          document.getElementById('pool-join-form').style.display = 'none';
        },
        'pool-join-btn': function () {
          document.getElementById('pool-join-form').style.display = '';
          document.getElementById('pool-create-form').style.display = 'none';
          var input = document.getElementById('pool-join-code');
          if (input) input.focus();
        },
        'pool-create-submit': function () { self.createPool(); },
        'pool-join-submit': function () { self.joinPool(); },
        'pool-invite-code-btn': function () { self.generateInviteCode(); },
        'pool-leave-btn': function () { self.leavePool(); },
        'pool-copy-code-btn': function () { self.copyInviteCode(); },
        'pool-save-name-btn': function () { self.saveDeviceName(); }
      };
      Object.keys(ids).forEach(function (id) {
        var el = document.getElementById(id);
        if (el) el.addEventListener('click', ids[id]);
      });

      // Enter key shortcuts
      var enterBind = { 'pool-join-code': 'joinPool', 'pool-create-name': 'createPool', 'pool-device-name-input': 'saveDeviceName' };
      Object.keys(enterBind).forEach(function (id) {
        var el = document.getElementById(id);
        if (el) el.addEventListener('keydown', function (e) {
          if (e.key === 'Enter') self[enterBind[id]]();
        });
      });

      // Setup wizard pool join button
      var setupPoolJoin = document.getElementById('setup-pool-join');
      if (setupPoolJoin) setupPoolJoin.addEventListener('click', function () {
        var input = document.getElementById('setup-pool-code');
        var code = input ? input.value.trim().toUpperCase() : '';
        var status = document.getElementById('setup-pool-status');
        if (!code || code.length !== 8) {
          if (status) { status.textContent = I18n.t('pool.code_invalid'); status.style.color = 'var(--red)'; }
          return;
        }
        U.submitCodeForm('/api/pool/join', code, status, {
          pendingMsg: I18n.t('pool.linking'),
          successMsg: I18n.t('pool.link_sent'),
          failMsg: I18n.t('pool.join_failed'),
          errorMsg: I18n.t('pool.failed_generic', { error: 'network error' }),
          onSuccess: function() { if (input) input.value = ''; }
        });
      });

      // Check pool state on init to show slave banner
      this.checkSlaveBanner();
    },

    load: async function () {
      try {
        // Get our node ID for owner detection
        if (!this._myNodeId) {
          var statsResult = await App.data.loadStats();
          if (statsResult && statsResult.stats) {
            this._myNodeId = statsResult.stats.node_id || null;
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
        // Update slave banner on dashboard
        this.updateSlaveBanner(data);
      } catch (e) {
      }
    },

    renderNoPool: function () {
      var noPool = document.getElementById('pool-no-pool');
      var active = document.getElementById('pool-active');
      if (noPool) noPool.style.display = '';
      if (active) active.style.display = 'none';
      document.getElementById('pool-create-form').style.display = 'none';
      document.getElementById('pool-join-form').style.display = 'none';
    },

    renderActivePool: function (data) {
      var noPool = document.getElementById('pool-no-pool');
      var active = document.getElementById('pool-active');
      if (noPool) noPool.style.display = 'none';
      if (active) active.style.display = '';

      // Pool name
      var nameEl = document.getElementById('pool-name');
      if (nameEl) nameEl.textContent = data.name || I18n.t('pool.default_name');

      // Role label
      var roleEl = document.getElementById('pool-role-label');
      if (roleEl) {
        if (this._isOwner) {
          roleEl.textContent = I18n.t('pool.role_owner');
          roleEl.style.color = 'var(--green)';
        } else {
          roleEl.textContent = I18n.t('pool.role_member');
          roleEl.style.color = 'var(--cyan)';
        }
      }

      // Show owner-only controls
      var inviteBtn = document.getElementById('pool-invite-code-btn');
      if (inviteBtn) inviteBtn.style.display = this._isOwner ? '' : 'none';
      var splitSection = document.getElementById('pool-split-section');
      if (splitSection) splitSection.style.display = this._isOwner ? '' : 'none';

      // Stats
      var members = data.members || [];
      var el = function (id) { return document.getElementById(id); };
      if (el('pool-member-count')) el('pool-member-count').textContent = members.length;
      if (el('pool-total-credits')) el('pool-total-credits').textContent = (data.total_lifetime_credits || 0).toLocaleString();

      // Aggregate VRAM
      var totalVram = 0;
      var onlineCount = 0;
      members.forEach(function (m) {
        if (m.stats && m.stats.vram_mb) totalVram += m.stats.vram_mb;
        if (m.online) onlineCount++;
      });
      if (el('pool-total-vram')) el('pool-total-vram').textContent = totalVram > 0 ? U.formatMB(totalVram) : '—';
      if (el('pool-online-count')) el('pool-online-count').textContent = onlineCount + '/' + members.length;



      // Device name input (my current name)
      var nameInput = document.getElementById('pool-device-name-input');
      if (nameInput) {
        var myMember = members.find(function (m) { return m.node_id === App.pool._myNodeId; });
        if (myMember && myMember.device_name) {
          nameInput.value = myMember.device_name;
        }
      }

      this.renderMembers(members);
    },

    renderMembers: function (members) {
      var list = document.getElementById('pool-members-list');
      if (!list) return;
      list.innerHTML = '';

      if (members.length === 0) {
        list.innerHTML = '<div class="empty-state text-muted">' +
          U.escapeHtml(I18n.t('pool.no_members')) + '</div>';
        return;
      }

      var tmpl = document.getElementById('tmpl-pool-member-row');
      if (!tmpl) return;

      var self = this;
      var poolId = this._poolState ? this._poolState.pool_id : null;

      members.forEach(function (m) {
        var row = tmpl.content.cloneNode(true).firstElementChild;

        var idEl = row.querySelector('.pool-member-id');
        var joinedEl = row.querySelector('.pool-member-joined');
        var creditsEl = row.querySelector('.pool-member-credits');
        var removeBtn = row.querySelector('.pool-member-remove-btn');
        var statusEl = row.querySelector('.pool-member-status');
        var statsEl = row.querySelector('.pool-member-stats-detail');
        var iconEl = row.querySelector('.pool-member-icon');

        var isSelf = m.node_id === self._myNodeId;
        var isOwnerDevice = m.node_id === poolId;

        // Device name or truncated ID
        var displayName = m.device_name || (m.node_id ? m.node_id.substring(0, 12) + '...' : '?');
        if (isSelf) displayName += I18n.t('pool.you_suffix');
        if (idEl) idEl.textContent = displayName;

        // Online status dot
        if (statusEl) {
          statusEl.style.color = m.online ? 'var(--green)' : 'var(--text-muted)';
          statusEl.textContent = m.online ? '\u25CF' : '\u25CB'; // filled/empty circle
          statusEl.title = m.online
            ? I18n.t('pool.online')
            : I18n.t('pool.offline') +
              (m.last_seen ? ' — ' + I18n.t('pool.last_seen') + ' ' + new Date(m.last_seen).toLocaleString() : '');
        }

        // Joined date
        if (joinedEl) joinedEl.textContent = m.joined_at ? m.joined_at.substring(0, 10) : '?';

        // Credits
        if (creditsEl) creditsEl.textContent = (m.credits_contributed || 0).toLocaleString();

        // Per-device stats (if available)
        if (statsEl && m.stats) {
          var parts = [];
          if (m.stats.vram_mb > 0) parts.push(I18n.t('pool.stats_vram', { size: U.formatMB(m.stats.vram_mb) }));
          if (m.stats.shards_hosted > 0) parts.push(I18n.t('pool.stats_shards', { n: m.stats.shards_hosted }));
          if (m.stats.forwards_served > 0) parts.push(I18n.t('pool.stats_forwards', { n: m.stats.forwards_served }));
          if (m.stats.uptime_secs > 0) parts.push(U.formatUptime(m.stats.uptime_secs));
          statsEl.textContent = parts.join(' · ') || '';
          statsEl.style.display = parts.length > 0 ? '' : 'none';
        }

        // Icon
        if (iconEl) {
          if (isOwnerDevice) iconEl.textContent = '\uD83D\uDC51'; // crown
          else if (isSelf) iconEl.textContent = '\u2B50'; // star
          else iconEl.textContent = '\uD83D\uDCBB'; // laptop
        }

        // Contribution level slider (owner only, not self/owner device)
        var contribSection = row.querySelector('.pool-member-contribution');
        if (contribSection && self._isOwner && !isOwnerDevice) {
          contribSection.style.display = '';
          var slider = contribSection.querySelector('.pool-contrib-slider');
          var label = contribSection.querySelector('.pool-contrib-label');
          if (slider) {
            slider.value = m.contribution_level || 100;
            if (label) label.textContent = (m.contribution_level || 100) + '%';
            slider.setAttribute('data-node-id', m.node_id);
            slider.addEventListener('input', function () {
              var lbl = this.parentElement.querySelector('.pool-contrib-label');
              if (lbl) lbl.textContent = this.value + '%';
            });
            slider.addEventListener('change', function () {
              var nid = this.getAttribute('data-node-id');
              App.pool.setContribution(nid, parseInt(this.value, 10));
            });
          }
        }

        // Remove button (owner only, not self)
        if (removeBtn && self._isOwner && !isSelf) {
          removeBtn.style.display = '';
          removeBtn.setAttribute('data-pool-remove', m.node_id);
          removeBtn.addEventListener('click', function () {
            self.removeMember(this.getAttribute('data-pool-remove'));
          });
        }

        list.appendChild(row);
      });
    },

    createPool: async function () {
      var input = document.getElementById('pool-create-name');
      var name = input ? input.value.trim() : '';
      if (!name) {
        App.notifications.showToast(I18n.t('pool.name_required'), 'error');
        return;
      }
      try {
        var resp = await App.authFetch('/api/pool/create', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ name: name })
        });
        var data = await resp.json();
        if (_hasError(data)) return;
        App.notifications.showToast(I18n.t('pool.created_success'), 'success');
        this.load();
      } catch (e) {
        App.notifications.showToast(I18n.t('pool.failed_generic', { error: e.message }), 'error');
      }
    },

    joinPool: async function () {
      var input = document.getElementById('pool-join-code');
      var code = input ? input.value.trim().toUpperCase() : '';
      if (!code || code.length !== 8) {
        App.notifications.showToast(I18n.t('pool.code_invalid'), 'error');
        return;
      }
      try {
        var resp = await App.authFetch('/api/pool/join', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ code: code })
        });
        var data = await resp.json();
        if (_hasError(data)) return;
        App.notifications.showToast(I18n.t('pool.join_sent'), 'success');
        if (input) input.value = '';
        setTimeout(function () { App.pool.load(); }, 5000);
        setTimeout(function () { App.pool.load(); }, 15000);
      } catch (e) {
        App.notifications.showToast(I18n.t('pool.failed_generic', { error: e.message }), 'error');
      }
    },

    generateInviteCode: async function () {
      try {
        var resp = await App.authFetch('/api/pool/generate-code', { method: 'POST' });
        var data = await resp.json();
        if (_hasError(data)) return;
        var code = data.code;
        var display = document.getElementById('pool-invite-code-display');
        var codeVal = document.getElementById('pool-invite-code-value');
        if (display) display.style.display = '';
        if (codeVal) codeVal.textContent = code;
        this._lastCode = code;

        // Also generate QR code
        this.renderQR(code);
      } catch (e) {
        App.notifications.showToast(I18n.t('pool.failed_generic', { error: e.message }), 'error');
      }
    },

    renderQR: function (code) {
      var container = document.getElementById('pool-qr-code');
      if (!container) return;
      container.innerHTML = '';
      container.style.display = '';

      // Simple QR code using a canvas-based generator
      // We use a minimal QR encoding — for 8 alphanumeric chars, version 1 is sufficient
      // Fallback: just show the code in a styled box if QR lib isn't available
      // Create a visual representation using CSS grid (works without any library)
      var size = 120;
      var canvas = document.createElement('canvas');
      canvas.width = size;
      canvas.height = size;
      canvas.style.borderRadius = '8px';
      canvas.style.border = '4px solid white';
      var ctx = canvas.getContext('2d');

      // Generate a simple visual pattern from the code hash (not a real QR, but distinctive)
      // Real QR would require a library — this is a recognizable visual shorthand
      ctx.fillStyle = '#ffffff';
      ctx.fillRect(0, 0, size, size);
      ctx.fillStyle = '#000000';

      // Use code bytes to create a unique grid pattern
      var gridSize = 11;
      var cellSize = Math.floor(size / (gridSize + 2));
      var offset = Math.floor((size - cellSize * gridSize) / 2);

      // QR-style finder patterns in corners
      var drawFinder = function (x, y) {
        for (var r = 0; r < 7; r++) {
          for (var c = 0; c < 7; c++) {
            var fill = (r === 0 || r === 6 || c === 0 || c === 6) ||
                       (r >= 2 && r <= 4 && c >= 2 && c <= 4);
            if (fill) {
              ctx.fillRect(offset + (x + c) * cellSize, offset + (y + r) * cellSize, cellSize, cellSize);
            }
          }
        }
      };
      drawFinder(0, 0);
      drawFinder(gridSize - 7, 0);
      drawFinder(0, gridSize - 7);

      // Data area — use code chars to fill
      for (var i = 0; i < code.length; i++) {
        var charCode = code.charCodeAt(i);
        for (var bit = 0; bit < 5; bit++) {
          if ((charCode >> bit) & 1) {
            var pos = i * 5 + bit;
            var row = 7 + Math.floor(pos / 4);
            var col = 7 + (pos % 4);
            if (row < gridSize && col < gridSize) {
              ctx.fillRect(offset + col * cellSize, offset + row * cellSize, cellSize, cellSize);
            }
          }
        }
      }

      container.appendChild(canvas);

      // Add "scan or type" label
      var label = document.createElement('div');
      label.className = 'text-muted';
      label.style.fontSize = '0.72rem';
      label.style.marginTop = '4px';
      label.textContent = I18n.t('pool.scan_or_type');
      container.appendChild(label);
    },

    copyInviteCode: function () {
      var code = this._lastCode;
      if (!code) return;
      navigator.clipboard.writeText(code).then(function () {
        App.notifications.showToast(I18n.t('pool.code_copied'), 'success');
      });
    },

    saveDeviceName: async function () {
      var input = document.getElementById('pool-device-name-input');
      var name = input ? input.value.trim() : '';
      try {
        var resp = await App.authFetch('/api/pool/device-name', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ name: name })
        });
        var data = await resp.json();
        if (_hasError(data)) return;
        App.notifications.showToast(I18n.t('pool.name_saved'), 'success');
        this.load();
      } catch (e) {
        App.notifications.showToast(I18n.t('pool.failed_generic', { error: e.message }), 'error');
      }
    },

    leavePool: async function () {
      if (!confirm(I18n.t('pool.confirm_leave'))) return;
      try {
        var resp = await App.authFetch('/api/pool/leave', { method: 'POST' });
        var data = await resp.json();
        if (_hasError(data)) return;
        App.notifications.showToast(I18n.t('pool.left'), 'success');
        this.load();
      } catch (e) {
        App.notifications.showToast(I18n.t('pool.failed_generic', { error: e.message }), 'error');
      }
    },

    removeMember: async function (nodeId) {
      if (!confirm(I18n.t('pool.confirm_remove'))) return;
      try {
        var resp = await App.authFetch('/api/pool/remove', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ node_id: nodeId })
        });
        var data = await resp.json();
        if (_hasError(data)) return;
        App.notifications.showToast(I18n.t('pool.member_removed'), 'success');
        this.load();
      } catch (e) {
        App.notifications.showToast(I18n.t('pool.failed_generic', { error: e.message }), 'error');
      }
    },

    setContribution: async function (nodeId, level) {
      try {
        var resp = await App.authFetch('/api/pool/contribution', {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ node_id: nodeId, level: level })
        });
        var data = await resp.json();
        _hasError(data);
      } catch (e) {
        App.notifications.showToast(I18n.t('pool.failed_generic', { error: e.message }), 'error');
      }
    },

    /// Check pool state on startup and show/hide the slave banner.
    /// Reuses load() to avoid duplicate /api/pool/state fetch.
    checkSlaveBanner: async function () {
      try {
        // Delay slightly to let the API key load
        await new Promise(function (r) { setTimeout(r, 2000); });
        await this.load();
        if (this._poolState) {
          this.updateSlaveBanner(this._poolState);
        }
      } catch (e) {
        // Silent — banner is secondary
      }
    },

    /// Show/hide the persistent slave banner at the top of every page.
    /// The full dashboard remains usable — inference charges go to the master.
    updateSlaveBanner: function (data) {
      var banner = document.getElementById('slave-top-banner');
      if (!banner) return;

      var isSlave = data && data.in_pool && this._myNodeId !== null && data.pool_id !== this._myNodeId;
      if (isSlave) {
        banner.classList.add('visible');
        // Update banner text with owner info
        var textEl = document.getElementById('slave-banner-text');
        if (textEl) {
          var members = data.members || [];
          var ownerMember = members.find(function (m) { return m.node_id === data.pool_id; });
          var ownerName = (ownerMember && ownerMember.device_name) || data.name || I18n.t('pool.main_device_fallback');
          textEl.textContent = I18n.t('pool.slave_banner_detail', { owner: ownerName });
        }
      } else {
        banner.classList.remove('visible');
      }
    }
  };
})();
