'use strict';

// ============================================================================
// SwarmLLM — Downloads + Prune Schedule Component
// Download queue, prune history, resource schedule
// ============================================================================

(function() {
  var S = App.state;
  var U = App.utils;

  // ========================================================================
  // Download Queue
  // ========================================================================
  App.downloads = {
    load: async function() {
      try {
        var resp = await App.authFetch('/api/admin/downloads');
        if (!resp.ok) return;
        var data = await resp.json();
        App.downloads.render(data.downloads || []);
      } catch (e) {
        // Download queue is non-critical
      }
    },

    renderItem: function(dl) {
      var tmpl = document.getElementById('tmpl-dl-queue-item');
      var node = tmpl.content.cloneNode(true);
      var item = node.querySelector('.dl-queue-item');
      item.setAttribute('data-dl-model', dl.model_id);

      var nameEl = item.querySelector('.dl-queue-name');
      nameEl.textContent = dl.model_name || dl.model_id;
      nameEl.title = dl.model_id;

      var sourceEl = item.querySelector('.dl-queue-source');
      sourceEl.textContent = dl.source === 'huggingface' ? 'HF' : 'Network';
      sourceEl.classList.add(dl.source === 'huggingface' ? 'hf' : 'net');

      var stateName = typeof dl.state === 'string' ? dl.state : 'unknown';
      var stateLabel = stateName, stateClass = 'waiting';
      if (stateName === 'downloading') { stateLabel = I18n.t('dl.state_downloading'); stateClass = 'active'; }
      else if (stateName === 'awaiting_manifest') { stateLabel = I18n.t('dl.state_preparing'); stateClass = 'waiting'; }
      else if (stateName === 'complete') { stateLabel = I18n.t('dl.state_complete'); stateClass = 'done'; }
      else if (stateName.indexOf('failed') >= 0 || typeof dl.state === 'object') {
        stateLabel = I18n.t('dl.state_failed'); stateClass = 'fail';
        if (typeof dl.state === 'object' && dl.state.failed) {
          stateLabel += ': ' + (dl.state.failed.reason || '').substring(0, 40);
        }
      }
      var stateEl = item.querySelector('.dl-queue-state');
      stateEl.textContent = stateLabel;
      stateEl.classList.add(stateClass);

      if (dl.cancellable) {
        var cancelBtn = document.createElement('button');
        cancelBtn.className = 'dl-queue-cancel';
        cancelBtn.setAttribute('data-dl-cancel', dl.model_id);
        cancelBtn.textContent = I18n.t('actions.cancel');
        item.querySelector('.dl-queue-actions').appendChild(cancelBtn);
      }

      var pct = dl.overall_pct != null ? dl.overall_pct :
        (dl.total_bytes > 0 ? Math.min(100, Math.round((dl.downloaded_bytes / dl.total_bytes) * 100)) : 0);
      item.querySelector('.dl-queue-bar-fill').style.width = pct + '%';

      var shardInfo = (dl.downloaded_shards || 0) + '/' + (dl.total_shards || 0) + ' shards';
      if (dl.verified_shards > 0) shardInfo += ' (' + dl.verified_shards + ' verified)';
      item.querySelector('.dqs-left').textContent = shardInfo + ' \u00b7 ' + pct + '%';

      var speed = dl.speed_bytes_per_sec || 0;
      var statsRight = U.formatBytes(dl.downloaded_bytes || 0) + ' / ' + U.formatBytes(dl.total_bytes || 0);
      if (speed > 0) statsRight += ' \u00b7 ' + U.formatSpeed(speed);
      if (dl.eta_secs) statsRight += ' \u00b7 ETA ' + U.formatEta(dl.eta_secs);
      item.querySelector('.dqs-right').textContent = statsRight;

      if (dl.log && dl.log.length > 0) {
        var logRow = item.querySelector('.dl-queue-log-row');
        var logToggle = document.createElement('button');
        logToggle.className = 'dl-queue-log-toggle';
        logToggle.setAttribute('data-dl-log-toggle', dl.model_id);
        logToggle.textContent = I18n.t('downloads.log_count', { count: dl.log.length });
        logRow.appendChild(logToggle);

        var logPanel = document.createElement('div');
        logPanel.className = 'dl-queue-log';
        logPanel.setAttribute('data-dl-log', dl.model_id);
        dl.log.forEach(function(l) {
          var line = document.createElement('div');
          line.className = 'dl-queue-log-line';
          line.textContent = l;
          logPanel.appendChild(line);
        });
        item.appendChild(logPanel);
      }

      return item;
    },

    render: function(downloads) {
      var panel = document.getElementById('download-queue-panel');
      var list = document.getElementById('download-queue-list');
      var empty = document.getElementById('download-queue-empty');
      var count = document.getElementById('download-queue-count');
      if (!panel || !list) return;

      var active = downloads.filter(function(d) {
        var st = typeof d.state === 'string' ? d.state : '';
        return st !== 'complete';
      });

      if (active.length === 0 && downloads.length === 0) { panel.classList.add('hidden'); return; }
      panel.classList.remove('hidden');
      if (active.length === 0) {
        list.innerHTML = '';
        if (empty) empty.classList.remove('hidden');
        if (count) count.textContent = '';
        return;
      }

      if (empty) empty.classList.add('hidden');
      if (count) count.textContent = I18n.t('downloads.active_count', { count: active.length });
      list.innerHTML = '';
      active.forEach(function(dl) {
        list.appendChild(App.downloads.renderItem(dl));
      });
    },

    updateFromWs: function(acquisitions) {
      if (!acquisitions || acquisitions.length === 0) return;
      var panel = document.getElementById('download-queue-panel');
      var list = document.getElementById('download-queue-list');
      if (!panel || !list) return;

      var hasActive = acquisitions.some(function(a) {
        var st = typeof a.state === 'string' ? a.state : '';
        return st === 'downloading' || st === 'awaiting_manifest';
      });

      if (hasActive && panel.classList.contains('hidden')) {
        App.downloads.render(acquisitions);
        return;
      }

      var count = document.getElementById('download-queue-count');
      acquisitions.forEach(function(acq) {
        var existing = list.querySelector('[data-dl-model="' + U.cssSafeAttr(acq.model_id) + '"]');

        if (!existing) {
          if (acq.state === 'downloading' || acq.state === 'awaiting_manifest') {
            panel.classList.remove('hidden');
            var empty = document.getElementById('download-queue-empty');
            if (empty) empty.classList.add('hidden');
            list.prepend(App.downloads.renderItem(acq));
            if (count) {
              var n = list.querySelectorAll('.dl-queue-item').length;
              count.textContent = I18n.t('downloads.active_suffix', { count: n });
            }
          }
          return;
        }

        var totalBytes = acq.total_bytes || 0;
        var dlBytes = acq.downloaded_bytes || 0;
        var pct = acq.overall_pct != null ? acq.overall_pct :
          (totalBytes > 0 ? Math.min(100, Math.round((dlBytes / totalBytes) * 100)) : 0);
        var speed = acq.speed_bytes_per_sec || 0;

        var barFill = existing.querySelector('.dl-queue-bar-fill');
        if (barFill) barFill.style.width = pct + '%';

        var leftEl = existing.querySelector('.dqs-left');
        if (leftEl) {
          var shardInfo = (acq.downloaded_shards || 0) + '/' + (acq.total_shards || 0) + ' shards';
          leftEl.textContent = shardInfo + ' \u00b7 ' + pct + '%';
        }

        var rightEl = existing.querySelector('.dqs-right');
        if (rightEl) {
          var right = U.formatBytes(dlBytes) + ' / ' + U.formatBytes(totalBytes);
          if (speed > 0) right += ' \u00b7 ' + U.formatSpeed(speed);
          if (acq.eta_secs) right += ' \u00b7 ETA ' + U.formatEta(acq.eta_secs);
          else if (speed > 0 && totalBytes > dlBytes) right += ' \u00b7 ETA ' + U.formatEta((totalBytes - dlBytes) / speed);
          rightEl.textContent = right;
        }

        if (typeof acq.state === 'string' && acq.state === 'complete') {
          setTimeout(function() { App.downloads.load(); }, 2000);
        }
      });
    },

    cancelDownload: function(modelId) {
      return App.models.cancelDownload(modelId);
    }
  };

  // ========================================================================
  // Prune History + Resource Schedule
  // ========================================================================
  function buildPruneRow(e) {
    var tmpl = document.getElementById('tmpl-prune-row');
    var row = tmpl.content.cloneNode(true).firstElementChild;
    row.querySelector('.prune-left').textContent = (e.model_name || e.model_id) + ' shard ' + e.shard_index;
    var freed = U.formatBytes(e.freed_bytes || 0);
    var ts = e.timestamp ? new Date(e.timestamp).toLocaleString() : '';
    row.querySelector('.prune-right').textContent = freed + ' \u2022 ' + e.holder_count_before + '\u2192' + e.holder_count_after + ' \u2022 ' + ts;
    return row;
  }

  function renderPruneHistory(events) {
    var el = document.getElementById('prune-history-list');
    if (!el) return;
    if (events.length === 0) {
      el.innerHTML = '<div class="text-muted" style="padding:0.5rem">' + U.escapeHtml(I18n.t('downloads.no_prune_events')) + '</div>';
      return;
    }
    el.innerHTML = '';
    events.slice(0, 20).forEach(function(e) {
      el.appendChild(buildPruneRow(e));
    });
  }

  function renderScheduleCard(s) {
    var el = document.getElementById('schedule-form');
    if (!el) return;
    el.innerHTML =
      '<div class="am-row"><label><input type="checkbox" id="sched-enabled"' + (s.enabled ? ' checked' : '') + '> ' + U.escapeHtml(I18n.t('downloads.enable_reduced')) + '</label></div>' +
      '<div class="am-row"><label>' + U.escapeHtml(I18n.t('downloads.start_hour')) + '</label> <input type="number" id="sched-start" value="' + (s.reduced_hours_start || 22) + '" min="0" max="23" style="width:3rem"></div>' +
      '<div class="am-row"><label>' + U.escapeHtml(I18n.t('downloads.end_hour')) + '</label> <input type="number" id="sched-end" value="' + (s.reduced_hours_end || 8) + '" min="0" max="23" style="width:3rem"></div>' +
      '<div class="am-row"><label>' + U.escapeHtml(I18n.t('downloads.contribution')) + '</label> <select id="sched-contrib"><option value="minimal"' + (s.reduced_contribution === 'minimal' ? ' selected' : '') + '>' + U.escapeHtml(I18n.t('downloads.level_minimal')) + '</option><option value="moderate"' + (s.reduced_contribution === 'moderate' ? ' selected' : '') + '>' + U.escapeHtml(I18n.t('downloads.level_moderate')) + '</option></select></div>' +
      '<div class="am-row"><label>' + U.escapeHtml(I18n.t('downloads.prune_aggressiveness')) + '</label> <select id="sched-prune-agg"><option value="conservative"' + (s.prune_aggressiveness === 'conservative' ? ' selected' : '') + '>' + U.escapeHtml(I18n.t('downloads.level_conservative')) + '</option><option value="normal"' + (s.prune_aggressiveness === 'normal' ? ' selected' : '') + '>' + U.escapeHtml(I18n.t('downloads.level_normal')) + '</option><option value="aggressive"' + (s.prune_aggressiveness === 'aggressive' ? ' selected' : '') + '>' + U.escapeHtml(I18n.t('downloads.level_aggressive')) + '</option></select></div>' +
      '<div class="am-row"><button class="btn btn-sm btn-primary" id="sched-save-btn">' + U.escapeHtml(I18n.t('downloads.save_schedule')) + '</button></div>';
    var saveBtn = document.getElementById('sched-save-btn');
    if (saveBtn) saveBtn.addEventListener('click', saveSchedule);
  }

  async function saveSchedule() {
    try {
      var resp = await App.authFetch('/api/admin/schedule', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          enabled: document.getElementById('sched-enabled').checked,
          reduced_hours_start: parseInt(document.getElementById('sched-start').value, 10) || 22,
          reduced_hours_end: parseInt(document.getElementById('sched-end').value, 10) || 8,
          reduced_contribution: document.getElementById('sched-contrib').value,
          prune_aggressiveness: document.getElementById('sched-prune-agg').value,
        }),
      });
      if (resp.ok) {
        App.ui.showBanner('success', I18n.t('downloads.schedule_saved'));
      } else {
        App.ui.showBanner('error', await U.getApiErrorMessage(resp, I18n.t('downloads.save_failed')));
      }
    } catch (e) {
      App.ui.showBanner('error', I18n.t('downloads.save_error', { error: e.message }));
    }
  }

  App.pruneSchedule = {
    loadHistory: async function() {
      try {
        var resp = await App.authFetch('/api/admin/prune-history');
        if (!resp.ok) return;
        var data = await resp.json();
        renderPruneHistory(data.events || []);
      } catch (e) {
        var el = document.getElementById('prune-history-list');
        if (el) el.innerHTML = '<div class="text-muted" style="padding:0.5rem">' + App.utils.escapeHtml(I18n.t('downloads.prune_load_error')) + '</div>';
      }
    },

    loadSchedule: async function() {
      var el = document.getElementById('schedule-form');
      try {
        var resp = await App.authFetch('/api/admin/schedule');
        if (!resp.ok) {
          if (el) el.innerHTML = '<div class="text-muted" style="font-size:0.85rem">' + App.utils.escapeHtml(I18n.t('downloads.no_schedule')) + '</div>';
          return;
        }
        var s = await resp.json();
        renderScheduleCard(s);
      } catch (e) {
        if (el) el.innerHTML = '<div class="text-muted" style="font-size:0.85rem">' + U.escapeHtml(I18n.t('downloads.no_schedule')) + '</div>';
      }
    },

    prependHistory: function(e) {
      var el = document.getElementById('prune-history-list');
      if (!el) return;
      var placeholder = el.querySelector('.text-muted');
      if (placeholder && el.children.length === 1) el.innerHTML = '';
      el.prepend(buildPruneRow(e));
      while (el.children.length > 20) el.removeChild(el.lastChild);
    }
  };
})();
