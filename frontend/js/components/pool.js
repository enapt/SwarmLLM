// ── Device Pool component ──
// Manages the "My Devices" tab: create/join pool, invite codes, member list,
// device nicknames, online status, per-device stats, leave.

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

  // Sniff v2 (`swarmpool://...`) vs legacy 8-char code and normalise.
  // Returns `null` when input is empty or fails the legacy charset check;
  // v2 blobs are case-preserved (base64url body is case-sensitive),
  // legacy codes are upper-cased. The two pool-join entry points
  // (setup-wizard click handler and `joinPool`) share this.
  function _normaliseCode(raw) {
    if (!raw) return null;
    var isV2 = /^swarmpool:\/\//i.test(raw);
    var code = isV2 ? raw : raw.toUpperCase();
    if (!code || (!isV2 && !/^[A-Z0-9]{8}$/.test(code))) return null;
    return { isV2: isV2, code: code };
  }

  async function _poolAction(url, opts, successMsg) {
    try {
      var resp = await App.authFetch(url, opts);
      var data = await resp.json();
      if (_hasError(data)) return;
      if (successMsg) App.notifications.showToast(successMsg, 'success');
      App.pool.load();
    } catch (e) {
      App.notifications.showToast(I18n.t('pool.failed_generic', { error: e.message }), 'error');
    }
  }

  function _jsonOpts(method, body) {
    return { method: method, headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) };
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
        'pool-invite-code-btn-settings': function () { self.generateInviteCode(); },
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

      // Setup wizard pool join button — accepts both swarmpool:// blob and
      // legacy 8-char code; only the legacy form gets case-normalized.
      var setupPoolJoin = document.getElementById('setup-pool-join');
      if (setupPoolJoin) setupPoolJoin.addEventListener('click', function () {
        var input = document.getElementById('setup-pool-code');
        var raw = input ? input.value.trim() : '';
        var status = document.getElementById('setup-pool-status');
        var parsed = _normaliseCode(raw);
        if (!parsed) {
          if (status) { status.textContent = I18n.t('pool.code_invalid'); status.style.color = 'var(--red)'; }
          return;
        }
        U.submitCodeForm('/api/pool/join', parsed.code, status, {
          pendingMsg: I18n.t('pool.linking'),
          successMsg: I18n.t('pool.link_sent'),
          failMsg: I18n.t('pool.join_failed'),
          errorMsg: I18n.t('pool.failed_generic', { error: I18n.t('common.request_failed') }),
          onSuccess: function() { if (input) input.value = ''; }
        });
      });

      // Check pool state on init to show slave banner
      this.checkSlaveBanner();
      // Wire private mode toggles
      this.initPrivateMode();
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
          this._privateMode = !!data.private_mode;
          this.renderActivePool(data);
          this.updatePrivateModeUI();
          if (this._privateMode) this.loadCoverage();
        } else {
          this._privateMode = false;
          this.updatePrivateModeUI();
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
      if (nameEl) nameEl.textContent = data.name || I18n.t('pool.title');

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

      // Show owner-only controls. The invite-code button has two homes:
      //   - Header (prominent): shown while the global swarm is still
      //     young (this node sees <50 connected peers). The whole point
      //     of the invite code is to bootstrap-before-decentralization,
      //     so during the growth phase it should be in everyone's face.
      //   - Settings section (demoted): shown once this node is well
      //     embedded in the swarm — at 50+ peers, Kademlia DHT discovery
      //     is reliable enough that new joiners can usually find this
      //     node without explicit rendezvous, so the prominent button
      //     stops serving a network-bootstrap mission. The owner can
      //     still mint codes anytime for personal-device pool joins;
      //     it's just demoted, not hidden.
      // Threshold of 50 matches the network-level maturity signal — NOT
      // the pool member count (pools cap at 10, threshold would never
      // trigger). Source: stats cache, populated by App.data.loadStats.
      var members = data.members || [];
      var MATURE_SWARM_THRESHOLD = 50;
      var statsCache = (App.data && App.data.cache && App.data.cache.stats) || null;
      // R140 maturity fade: stats serializes connected count as `peers`
      // (admin.rs and websocket.rs both — `peer_count` was a different
      // endpoint and gave undefined here, leaving the prominent button
      // up forever regardless of swarm size).
      var connectedPeers = statsCache ? (statsCache.peers || 0) : 0;
      var swarmIsMature = connectedPeers >= MATURE_SWARM_THRESHOLD;
      var headerInvite = document.getElementById('pool-invite-code-btn');
      var settingsInvite = document.getElementById('pool-settings-invite-section');
      if (headerInvite) headerInvite.style.display = (this._isOwner && !swarmIsMature) ? '' : 'none';
      if (settingsInvite) settingsInvite.style.display = (this._isOwner && swarmIsMature) ? '' : 'none';
      var splitSection = document.getElementById('pool-split-section');
      if (splitSection) splitSection.style.display = this._isOwner ? '' : 'none';

      // Stats
      var el = function (id) { return document.getElementById(id); };
      if (el('pool-member-count')) el('pool-member-count').textContent = members.length;

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
        var removeBtn = row.querySelector('.pool-member-remove-btn');
        var statusEl = row.querySelector('.pool-member-status');
        var statsEl = row.querySelector('.pool-member-stats-detail');
        var iconEl = row.querySelector('.pool-member-icon');
        var roleBadge = row.querySelector('.pool-member-role-badge');

        var isSelf = m.node_id === self._myNodeId;
        var isOwnerDevice = m.node_id === poolId;

        // Device name or truncated ID
        var displayName = m.device_name || (m.node_id ? m.node_id.substring(0, 12) + '...' : '?');
        if (isSelf) displayName += I18n.t('pool.you_suffix');
        if (idEl) idEl.textContent = displayName;

        // Role badge: Master vs Linked
        if (roleBadge) {
          if (isOwnerDevice) {
            roleBadge.textContent = I18n.t('pool.role_master');
            roleBadge.style.background = 'var(--green)';
            roleBadge.style.color = 'var(--on-solid)';
          } else {
            roleBadge.textContent = I18n.t('pool.role_linked');
            roleBadge.style.background = 'var(--border)';
            roleBadge.style.color = 'var(--text-primary)';
          }
        }

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
        if (joinedEl) joinedEl.textContent = m.joined_at
          ? I18n.t('pool.joined_prefix') + ' ' + m.joined_at.substring(0, 10)
          : '?';

        // Per-device stats (if available)
        if (statsEl && m.stats) {
          var parts = [];
          if (m.stats.vram_mb > 0) parts.push(I18n.t('pool.stats_vram', { size: U.formatMB(m.stats.vram_mb) }));
          if (m.stats.ram_mb > 0) parts.push(I18n.t('pool.stats_ram', { size: U.formatMB(m.stats.ram_mb) }));
          if (m.stats.shards_hosted > 0) parts.push(I18n.t('pool.stats_shards', { n: m.stats.shards_hosted }));
          if (m.stats.requests_served > 0) parts.push(I18n.t('pool.stats_requests', { n: m.stats.requests_served }));
          if (m.stats.forwards_served > 0) parts.push(I18n.t('pool.stats_forwards', { n: m.stats.forwards_served }));
          if (m.stats.uptime_secs > 0) parts.push(U.formatUptime(m.stats.uptime_secs));
          statsEl.textContent = parts.join(' \u00B7 ') || '';
          statsEl.style.display = parts.length > 0 ? '' : 'none';

          // Show models hosted in a sub-line
          if (m.stats.models_hosted && m.stats.models_hosted.length > 0) {
            var modelsLine = document.createElement('div');
            modelsLine.className = 'text-muted';
            modelsLine.style.cssText = 'font-size:0.78em;margin-top:2px;opacity:0.8;';
            modelsLine.textContent = I18n.t('pool.stats_models', { models: m.stats.models_hosted.join(', ') });
            statsEl.parentElement.insertBefore(modelsLine, statsEl.nextSibling);
          }
        } else if (statsEl) {
          statsEl.textContent = I18n.t('pool.stats_pending');
          statsEl.style.display = '';
          statsEl.style.opacity = '0.5';
          statsEl.style.fontStyle = 'italic';
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
      await _poolAction('/api/pool/create', _jsonOpts('POST', { name: name }), I18n.t('pool.created_success'));
    },

    joinPool: async function () {
      var input = document.getElementById('pool-join-code');
      var raw = input ? input.value.trim() : '';
      var parsed = _normaliseCode(raw);
      if (!parsed) {
        App.notifications.showToast(I18n.t('pool.code_invalid'), 'error');
        return;
      }
      try {
        var resp = await App.authFetch('/api/pool/join', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ code: parsed.code })
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
        if (display) display.classList.remove('hidden');
        if (codeVal) codeVal.textContent = code;
        this._lastCode = code;
      } catch (e) {
        App.notifications.showToast(I18n.t('pool.failed_generic', { error: e.message }), 'error');
      }
    },

    copyInviteCode: function () {
      var code = this._lastCode;
      if (!code) return;
      U.copyToClipboard(code, {
        onSuccess: function () {
          App.notifications.showToast(I18n.t('pool.code_copied'), 'success');
        },
      });
    },

    saveDeviceName: async function () {
      var input = document.getElementById('pool-device-name-input');
      var name = input ? input.value.trim() : '';
      await _poolAction('/api/pool/device-name', _jsonOpts('POST', { name: name }), I18n.t('pool.name_saved'));
    },

    leavePool: async function () {
      if (!confirm(I18n.t('pool.confirm_leave'))) return;
      await _poolAction('/api/pool/leave', { method: 'POST' }, I18n.t('pool.left'));
    },

    removeMember: async function (nodeId) {
      if (!confirm(I18n.t('pool.confirm_remove'))) return;
      await _poolAction('/api/pool/remove', _jsonOpts('POST', { node_id: nodeId }), I18n.t('pool.member_removed'));
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
    },

    // ---- Private Mode ----

    _privateMode: false,

    initPrivateMode: function () {
      var self = this;
      // Pool section toggle
      var checkbox = document.getElementById('pool-private-mode-checkbox');
      if (checkbox) {
        checkbox.addEventListener('change', function () {
          var cb = this;
          self.confirmPrivateMode(cb.checked, function () {
            // Revert checkbox if user cancels
            cb.checked = self._privateMode;
          });
        });
      }
      // Header shield button
      var headerBtn = document.getElementById('btn-private-mode-toggle');
      if (headerBtn) {
        headerBtn.addEventListener('click', function () {
          self.confirmPrivateMode(!self._privateMode);
        });
      }
    },

    confirmPrivateMode: async function (enabled, onCancel) {
      // Fetch coverage preview before confirming
      var coverageText = '';
      try {
        var resp = await App.authFetch('/api/pool/coverage');
        if (resp.ok) {
          var cov = await resp.json();
          coverageText = '\n\n' + I18n.t('pool.coverage_summary', {
            full: cov.fully_covered || 0,
            partial: cov.partially_covered || 0,
            none: cov.not_covered || 0
          });
          if (cov.est_total_download_mb > 0 && enabled) {
            coverageText += '\n' + I18n.t('pool.coverage_download_warning', {
              size: U.formatMB(cov.est_total_download_mb)
            });
          }
        }
      } catch (e) { /* ignore */ }

      var msg = enabled
        ? I18n.t('pool.confirm_enable_private') + coverageText
        : I18n.t('pool.confirm_disable_private');

      if (!confirm(msg)) {
        if (onCancel) onCancel();
        return;
      }
      this.setPrivateMode(enabled);
    },

    setPrivateMode: async function (enabled) {
      try {
        var resp = await App.authFetch('/api/pool/private-mode', {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ enabled: enabled })
        });
        if (!resp.ok) {
          var err = await resp.json().catch(function () { return {}; });
          App.notifications.showToast(
            (err.error && err.error.message) || I18n.t('pool.private_mode_error'),
            'error'
          );
          return;
        }
        var data = await resp.json();
        this._privateMode = data.enabled;
        this.updatePrivateModeUI();
        if (data.coverage) this.renderCoverage(data.coverage);
        App.notifications.showToast(
          data.enabled ? I18n.t('pool.private_mode_enabled') : I18n.t('pool.private_mode_disabled'),
          data.enabled ? 'info' : 'success'
        );
      } catch (e) {
        App.notifications.showToast(I18n.t('pool.private_mode_error'), 'error');
      }
    },

    updatePrivateModeUI: function () {
      var enabled = this._privateMode;
      // Pool section toggle
      var checkbox = document.getElementById('pool-private-mode-checkbox');
      if (checkbox) checkbox.checked = enabled;
      var label = document.getElementById('pool-private-mode-label');
      if (label) {
        label.textContent = enabled ? I18n.t('pool.private_mode_on') : I18n.t('pool.private_mode_off');
        label.style.color = enabled ? 'var(--green)' : '';
      }
      // Header shield
      var icon = document.getElementById('private-mode-icon');
      if (icon) {
        icon.style.color = enabled ? 'var(--green)' : '';
        icon.style.fill = enabled ? 'var(--green)' : 'none';
      }
      var badge = document.getElementById('private-mode-badge');
      if (badge) {
        if (enabled) badge.classList.remove('hidden');
        else badge.classList.add('hidden');
      }
      // Coverage panel
      var coveragePanel = document.getElementById('pool-coverage-panel');
      if (coveragePanel) coveragePanel.style.display = enabled ? '' : 'none';
    },

    loadCoverage: async function () {
      if (!this._privateMode) return;
      try {
        var resp = await App.authFetch('/api/pool/coverage');
        if (!resp.ok) return;
        var data = await resp.json();
        this.renderCoverage(data);
      } catch (e) { /* ignore */ }
    },

    renderCoverage: function (data) {
      var list = document.getElementById('pool-coverage-list');
      var summary = document.getElementById('pool-coverage-summary');
      if (!list) return;

      // Summary
      if (summary) {
        summary.textContent = I18n.t('pool.coverage_summary', {
          full: data.fully_covered || 0,
          partial: data.partially_covered || 0,
          none: data.not_covered || 0
        });
      }

      list.innerHTML = '';

      // Disk usage bar
      if (data.disk_budget_mb && data.disk_budget_mb > 0) {
        var usedPct = Math.min(100, Math.round((data.disk_used_mb || 0) / data.disk_budget_mb * 100));
        var diskColor = usedPct > 90 ? 'var(--red)' : usedPct > 70 ? 'var(--orange)' : 'var(--green)';
        var diskRow = document.createElement('div');
        diskRow.style.cssText = 'padding:8px 0;border-bottom:1px solid var(--border);margin-bottom:4px';
        diskRow.innerHTML =
          '<div style="display:flex;justify-content:space-between;font-size:0.78rem;margin-bottom:4px">' +
            '<span>' + I18n.t('pool.disk_usage') + '</span>' +
            '<span class="text-muted">' + U.formatMB(data.disk_used_mb || 0) + ' / ' + U.formatMB(data.disk_budget_mb) + '</span>' +
          '</div>' +
          '<div class="coverage-bar"><div class="coverage-bar-fill" style="width:' + usedPct + '%;background:' + diskColor + '"></div></div>';
        list.appendChild(diskRow);
      }

      var models = data.models || [];
      if (models.length === 0) {
        list.innerHTML += '<div class="text-muted" style="padding:8px">' +
          U.escapeHtml(I18n.t('pool.no_models_coverage')) + '</div>';
        return;
      }

      // Sort: fully covered first, then by coverage desc
      models.sort(function (a, b) {
        return b.coverage_pct - a.coverage_pct;
      });

      models.forEach(function (m) {
        var row = document.createElement('div');
        row.style.cssText = 'display:flex;align-items:center;gap:10px;padding:6px 0;border-bottom:1px solid var(--border)';

        var barColor = m.coverage_pct === 100 ? 'var(--green)'
          : m.coverage_pct > 0 ? 'var(--orange)' : 'var(--red)';

        var statusIcon = m.coverage_pct === 100 ? '\u2705'
          : m.coverage_pct > 0 ? '\u26A0\uFE0F' : '\u274C';

        row.innerHTML =
          '<span style="font-size:0.9rem;width:20px;text-align:center">' + statusIcon + '</span>' +
          '<div style="flex:1;min-width:0">' +
            '<div style="font-size:0.82rem;font-weight:500;white-space:nowrap;overflow:hidden;text-overflow:ellipsis">' +
              U.escapeHtml(m.name || m.id) + '</div>' +
            '<div style="display:flex;align-items:center;gap:8px;margin-top:2px">' +
              '<div class="coverage-bar" style="flex:1">' +
                '<div class="coverage-bar-fill" style="width:' + m.coverage_pct + '%;background:' + barColor + '"></div>' +
              '</div>' +
              '<span class="text-muted" style="font-size:0.7rem;white-space:nowrap">' +
                m.pool_shards + '/' + m.total_shards +
              '</span>' +
            '</div>' +
          '</div>' +
          (m.est_download_mb > 0
            ? '<span class="text-muted" style="font-size:0.7rem;white-space:nowrap">' +
                U.formatMB(m.est_download_mb) + ' ' + I18n.t('pool.coverage_needed') +
              '</span>'
            : '');

        list.appendChild(row);
      });
    }
  };
})();
