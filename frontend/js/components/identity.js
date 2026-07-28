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

    // Last fetched entries, so changing a filter re-renders without refetching.
    _lbEntries: [],

    loadLeaderboard: async function() {
      var tbody = document.getElementById('leaderboard-body');
      if (!tbody) return;
      try {
        var resp = await App.authFetch('/api/identity/leaderboard?limit=50');
        if (!resp.ok) {
          tbody.innerHTML = '<tr><td colspan="8" class="text-muted" style="text-align:center">' + U.escapeHtml(I18n.t('leaderboard.load_failed')) + '</td></tr>';
          return;
        }
        var data = await resp.json();
        App.identity._lbEntries = data.leaderboard || [];
        App.identity._populateLeaderboardFilters();
        App.identity.renderLeaderboard();
      } catch (e) {
        tbody.innerHTML = '<tr><td colspan="8" class="text-muted" style="text-align:center">' + U.escapeHtml(I18n.t('leaderboard.load_error', { error: e.message })) + '</td></tr>';
      }
    },

    // Filter options come from the data, not a hardcoded list: the set of
    // regions and systems on the network is whatever has actually joined it.
    _populateLeaderboardFilters: function() {
      var entries = App.identity._lbEntries;
      var fill = function(id, values) {
        var sel = document.getElementById(id);
        if (!sel) return;
        var current = sel.value;
        while (sel.options.length > 1) sel.remove(1);
        values.sort().forEach(function(v) {
          var o = document.createElement('option');
          o.value = v;
          o.textContent = v;
          sel.appendChild(o);
        });
        // Preserve the user's selection across a refresh unless the value is
        // gone, in which case fall back to All rather than silently filtering
        // to nothing.
        sel.value = values.indexOf(current) >= 0 ? current : '';
      };
      var regions = [], systems = [];
      entries.forEach(function(e) {
        var c = e.capability || {};
        if (c.region && regions.indexOf(c.region) < 0) regions.push(c.region);
        if (c.os && systems.indexOf(c.os) < 0) systems.push(c.os);
      });
      fill('lb-filter-region', regions);
      fill('lb-filter-os', systems);
    },

    _lbFilterValue: function(id) {
      var el = document.getElementById(id);
      return el ? el.value : '';
    },

    // A machine's one-line description. Falls back through what is known
    // rather than inventing anything: a node that predates a field renders as
    // unknown, never as a guess.
    _rigLabel: function(cap) {
      if (!cap || !cap.known) return I18n.t('leaderboard.unknown');
      if (cap.gpu && cap.gpu.name) {
        var vram = cap.gpu.vram_mb ? ' · ' + U.formatMB(cap.gpu.vram_mb) : '';
        return cap.gpu.name + vram;
      }
      return I18n.t('leaderboard.cpu_only') + (cap.ram_total_mb ? ' · ' + U.formatMB(cap.ram_total_mb) : '');
    },

    renderLeaderboard: function() {
      var tbody = document.getElementById('leaderboard-body');
      var podium = document.getElementById('leaderboard-podium');
      if (!tbody) return;

      var accel = App.identity._lbFilterValue('lb-filter-accel');
      var os = App.identity._lbFilterValue('lb-filter-os');
      var region = App.identity._lbFilterValue('lb-filter-region');

      var entries = App.identity._lbEntries.filter(function(e) {
        var c = e.capability || {};
        if (accel && c.accelerator !== accel) return false;
        if (os && c.os !== os) return false;
        if (region && c.region !== region) return false;
        return true;
      });

      var countEl = document.getElementById('lb-count');
      if (countEl) {
        countEl.textContent = I18n.t('leaderboard.showing', {
          shown: entries.length,
          total: App.identity._lbEntries.length
        });
      }

      // Podium only when the podium is complete AND unfiltered. Under a filter
      // "1st" would mean "first among CPU nodes in Germany", which the medal
      // does not say — so show the plain table instead of implying otherwise.
      var unfiltered = !accel && !os && !region;
      if (podium) {
        podium.innerHTML = '';
        if (unfiltered && entries.length >= 3) {
          podium.hidden = false;
          // Rank shown as a styled numeral, not a medal emoji: emoji fonts are
          // absent on plenty of Linux desktops and a tofu box reads as broken
          // UI. Colour carries gold/silver/bronze; the numeral carries the
          // meaning even with no colour perception at all.
          var tmpl = document.getElementById('tmpl-leaderboard-podium');
          for (var p = 0; p < 3; p++) {
            var pe = entries[p];
            var el = tmpl.content.cloneNode(true).firstElementChild;
            el.classList.add('lb-place-' + (p + 1));
            el.querySelector('.lb-medal').textContent = String(p + 1);
            el.querySelector('.lb-place-name').textContent = pe.display_name;
            el.querySelector('.lb-place-name').title = pe.node_id;
            el.querySelector('.lb-place-rig').textContent = App.identity._rigLabel(pe.capability);
            el.querySelector('.lb-place-credits').textContent =
              (pe.credits === null || pe.credits === undefined) ? '\u2014' : pe.credits;
            var pTier = el.querySelector('.lb-place-tier .tier-badge');
            var pTierName = pe.tier || I18n.t('leaderboard.tier_default');
            pTier.className = 'tier-badge ' + String(pTierName).toLowerCase().replace(/[^a-z]/g, '');
            pTier.textContent = pTierName;
            podium.appendChild(el);
          }
        } else {
          podium.hidden = true;
        }
      }

      if (entries.length === 0) {
        tbody.innerHTML = '<tr><td colspan="8" class="text-muted" style="text-align:center;padding:24px">' +
          U.escapeHtml(I18n.t(App.identity._lbEntries.length ? 'leaderboard.no_match' : 'leaderboard.empty')) + '</td></tr>';
        return;
      }

      tbody.innerHTML = '';
      var rowTmpl = document.getElementById('tmpl-leaderboard-row');
      for (var i = 0; i < entries.length; i++) {
        var e = entries[i];
        var c = e.capability || {};
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
        if (e.is_self) {
          var you = document.createElement('span');
          you.className = 'lb-you';
          you.textContent = I18n.t('leaderboard.you');
          nameCell.appendChild(you);
        }

        var rigCell = row.querySelector('.lb-rig');
        if (c.known) {
          var badge = document.createElement('span');
          badge.className = 'lb-accel lb-accel-' + (c.accelerator || 'cpu');
          badge.textContent = I18n.t(c.accelerator === 'gpu' ? 'leaderboard.gpu' : 'leaderboard.cpu');
          rigCell.appendChild(badge);
        }
        rigCell.appendChild(document.createTextNode(App.identity._rigLabel(c)));
        if (c.os) rigCell.title = c.os;

        row.querySelector('.lb-region').textContent = c.region || '\u2014';
        row.querySelector('.lb-speed').textContent =
          c.est_tokens_per_sec_7b ? c.est_tokens_per_sec_7b.toFixed(1) : '\u2014';
        row.querySelector('.lb-shards').textContent =
          (c.shards_hosted === undefined || c.shards_hosted === null) ? '\u2014' : c.shards_hosted;

        // `credits: null` means this peer's balance gossip hasn't arrived.
        // Show an em dash — NOT 0, which would read as a real balance.
        var creditsEl = row.querySelector('.lb-credits');
        var tierEl = row.querySelector('.lb-tier');
        if (e.credits === null || e.credits === undefined) {
          creditsEl.textContent = '\u2014';
          creditsEl.title = I18n.t('leaderboard.balance_unknown');
          tierEl.className = 'tier-badge';
          tierEl.textContent = '\u2014';
          tierEl.title = I18n.t('leaderboard.balance_unknown');
        } else {
          creditsEl.textContent = e.credits;
          var tierClass = (e.tier || 'silver').toLowerCase().replace(/[^a-z]/g, '');
          tierEl.className = 'tier-badge ' + tierClass;
          tierEl.textContent = e.tier || I18n.t('leaderboard.tier_default');
        }
        tbody.appendChild(row);
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
