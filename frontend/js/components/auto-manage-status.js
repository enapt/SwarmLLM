// Auto-manage header status indicator.
// Derives state from:
//   - stats_update acquisitions (active downloads)
//   - activity_event category=auto_manage (pruning, failures, cycle events)
//   - /api/admin/config (enabled flag, refreshed when settings save)
(function(App) {
  'use strict';
  if (!App) return;

  var PROBLEM_KINDS = {
    shard_verification_failed: 1,
    header_download_failed: 1,
    tied_output_failed: 1,
    mmproj_download_failed: 1,
    unsupported_architecture: 1,
    budget_exhausted: 1,
  };
  // Recent failure lingers for 60s before the orb clears back to idle.
  var PROBLEM_LINGER_MS = 60000;
  // Active prune flash lingers for 10s after the last shard_pruned event.
  var PRUNE_LINGER_MS = 10000;

  var state = {
    enabled: true,
    activeDownloads: 0,
    lastPruneTs: 0,
    lastProblem: null, // { kind, message, ts }
    lastCycleMsg: null,
  };

  function el(id) { return document.getElementById(id); }

  function setEnabled(on) {
    state.enabled = !!on;
    render();
  }

  function updateFromStats(acquisitions) {
    var active = 0;
    if (Array.isArray(acquisitions)) {
      acquisitions.forEach(function(a) {
        if (!a || !a.state) return;
        // Count in-progress downloads: state is an object {Downloading: {...}} or string "Downloading"
        var s = a.state;
        if (typeof s === 'string') {
          if (s === 'Downloading' || s === 'Queued' || s === 'Verifying') active++;
        } else if (typeof s === 'object') {
          if ('Downloading' in s || 'Queued' in s || 'Verifying' in s) active++;
        }
      });
    }
    state.activeDownloads = active;
    render();
  }

  function onEvent(data) {
    if (!data || data.category !== 'auto_manage') return;
    if (PROBLEM_KINDS[data.kind]) {
      state.lastProblem = {
        kind: data.kind,
        message: data.message || data.kind,
        ts: Date.now(),
      };
    }
    if (data.kind === 'shard_pruned') {
      state.lastPruneTs = Date.now();
    }
    if (data.kind === 'cycle_complete' || data.kind === 'interval_changed'
        || data.kind === 'hf_source_discovered' || data.kind === 'model_promoted'
        || data.kind === 'model_unloaded' || data.kind === 'vram_soft_unload') {
      state.lastCycleMsg = data.message || data.kind;
    }
    render();
  }

  function pickState() {
    var now = Date.now();
    if (!state.enabled) {
      return { cls: 'idle-disabled', label: '', titleKey: 'nav.auto_manage_idle_disabled',
               fallback: 'Auto-manage: disabled' };
    }
    if (state.lastProblem && (now - state.lastProblem.ts) < PROBLEM_LINGER_MS) {
      return { cls: 'problem', label: '!', titleKey: 'nav.auto_manage_problem',
               fallback: 'Auto-manage: ' + state.lastProblem.message, detail: state.lastProblem.message };
    }
    if (state.activeDownloads > 0) {
      var n = state.activeDownloads;
      return { cls: 'active-download', label: String(n),
               titleKey: 'nav.auto_manage_active_download',
               titleParams: { count: n },
               fallback: 'Auto-manage: downloading ' + n + ' ' + (n === 1 ? 'shard' : 'shards') };
    }
    if (state.lastPruneTs && (now - state.lastPruneTs) < PRUNE_LINGER_MS) {
      return { cls: 'active-prune', label: '', titleKey: 'nav.auto_manage_active_prune',
               fallback: 'Auto-manage: pruning' };
    }
    return { cls: 'idle-ok', label: '', titleKey: 'nav.auto_manage_idle_ok',
             fallback: 'Auto-manage: idle, healthy' };
  }

  var _renderTimer = null;

  function render() {
    var dot = el('auto-manage-dot');
    var btn = el('btn-auto-manage-status');
    var label = el('auto-manage-label');
    if (!dot || !btn || !label) return;

    var s = pickState();
    dot.className = s.cls;

    if (s.label) {
      label.textContent = s.label;
      label.classList.remove('hidden');
    } else {
      label.textContent = '';
      label.classList.add('hidden');
    }

    var title = (typeof I18n !== 'undefined') ? I18n.t(s.titleKey, s.titleParams || undefined) : s.fallback;
    if (!title || title === s.titleKey) title = s.fallback;
    if (s.detail && s.cls === 'problem') title += ' — ' + s.detail;
    btn.title = title;
    btn.setAttribute('aria-label', title);

    // Re-render when linger windows expire
    if (_renderTimer) { clearTimeout(_renderTimer); _renderTimer = null; }
    var now = Date.now();
    var nextIn = null;
    if (s.cls === 'problem') nextIn = PROBLEM_LINGER_MS - (now - state.lastProblem.ts) + 100;
    else if (s.cls === 'active-prune') nextIn = PRUNE_LINGER_MS - (now - state.lastPruneTs) + 100;
    if (nextIn && nextIn > 0) _renderTimer = setTimeout(render, nextIn);
  }

  function refreshEnabled() {
    if (!App.data || !App.data.loadConfig) return;
    App.data.loadConfig().then(function(cfg) {
      if (cfg && typeof cfg.auto_manage_shards === 'boolean') {
        setEnabled(cfg.auto_manage_shards);
      }
    }).catch(function() {});
  }

  function init() {
    refreshEnabled();
    render();
    var btn = el('btn-auto-manage-status');
    if (btn) {
      btn.addEventListener('click', function() {
        // Clicking the orb opens the settings dialog to the auto-manage section
        var s = el('btn-open-settings');
        if (s) s.click();
      });
    }
  }

  App.autoManageStatus = {
    init: init,
    setEnabled: setEnabled,
    refreshEnabled: refreshEnabled,
    updateFromStats: updateFromStats,
    onEvent: onEvent,
  };
})(window.App);
