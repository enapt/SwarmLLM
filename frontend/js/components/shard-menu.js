'use strict';

// ============================================================================
// SwarmLLM — Shard Context Menu Component
// Per-shard load/unload/delete/lock operations (extracted from models.js)
// ============================================================================

(function() {
  var U = App.utils;

  // ========================================================================
  // Shard Context Menu
  // ========================================================================
  App.shardMenu = {
    menu: null,
    currentModel: null,
    currentIndex: null,
    currentState: null,
    currentLocked: false,

    init: function() {
      this.menu = document.getElementById('shard-context-menu');
      // Wire up buttons once
      var loadBtn = document.getElementById('shard-ctx-load');
      if (loadBtn) loadBtn.addEventListener('click', function() { App.shardMenu.loadShard(); });
      var unloadBtn = document.getElementById('shard-ctx-unload');
      if (unloadBtn) unloadBtn.addEventListener('click', function() { App.shardMenu.unloadShard(); });
      var lockBtn = document.getElementById('shard-ctx-lock');
      if (lockBtn) lockBtn.addEventListener('click', function() { App.shardMenu.toggleLock(); });
    },

    show: function(modelId, shardIndex, shardState, x, y, isLocked, isInVram) {
      if (!this.menu) this.init();
      this.currentModel = modelId;
      this.currentIndex = shardIndex;
      this.currentState = shardState;
      this.currentLocked = !!isLocked;
      this.currentInVram = !!isInVram;

      var header = document.getElementById('shard-ctx-header');
      var statusEl = document.getElementById('shard-ctx-status');
      var btn = document.getElementById('shard-ctx-action');
      var unloadBtn = document.getElementById('shard-ctx-unload');
      var lockBtn = document.getElementById('shard-ctx-lock');
      var warnEl = document.getElementById('shard-ctx-warn');

      header.textContent = I18n.t('shard.part_n', { n: shardIndex + 1 });

      // Status line
      var statusText = '';
      if (shardState === 'local' && isInVram) statusText = I18n.t('shard.status_active');
      else if (shardState === 'local') statusText = I18n.t('shard.status_on_disk');
      else if (shardState === 'downloading') statusText = I18n.t('shard.status_downloading');
      else if (shardState === 'peer') statusText = I18n.t('shard.status_peer');
      else statusText = I18n.t('shard.status_unavailable');
      statusEl.textContent = statusText;

      // Primary action
      if (shardState === 'local') {
        btn.textContent = I18n.t('shard.delete');
        btn.className = 'shard-ctx-btn danger';
      } else if (shardState === 'downloading') {
        btn.textContent = I18n.t('shard.cancel_download');
        btn.className = 'shard-ctx-btn danger';
      } else {
        btn.textContent = I18n.t('shard.download');
        btn.className = 'shard-ctx-btn';
      }

      // Load button — only for local shards NOT in memory
      var loadBtn = document.getElementById('shard-ctx-load');
      if (loadBtn) {
        loadBtn.style.display = (shardState === 'local' && !isInVram) ? '' : 'none';
        loadBtn.title = 'Load this part into memory for inference. The model worker will restart to include it.';
      }

      // Unload button — only when shard is loaded in memory
      if (unloadBtn) {
        unloadBtn.style.display = (shardState === 'local' && isInVram) ? '' : 'none';
        unloadBtn.title = 'Keeps the file on disk but frees RAM/VRAM. The model worker will restart without this part.';
      }

      // Lock button — only for local shards
      if (lockBtn) {
        lockBtn.textContent = isLocked ? I18n.t('shard.unlock') : I18n.t('shard.lock');
        lockBtn.style.display = (shardState === 'local') ? '' : 'none';
      }

      // Warning when auto-manage is on
      if (warnEl) {
        warnEl.style.display = 'none';
        if (shardState === 'local') {
          warnEl.innerHTML = I18n.t('shard.auto_manage_warn');
          warnEl.style.display = '';
        }
      }

      var mw = 220, mh = 160;
      var left = Math.min(x, window.innerWidth - mw - 8);
      var top = Math.min(y, window.innerHeight - mh - 8);
      this.menu.style.left = left + 'px';
      this.menu.style.top = top + 'px';
      this.menu.style.display = '';
    },

    hide: function() {
      if (this.menu) this.menu.style.display = 'none';
    },

    execute: async function() {
      var modelId = this.currentModel;
      var idx = this.currentIndex;
      var state = this.currentState;
      this.hide();

      if (state === 'local') {
        if (!confirm(I18n.t('actions.confirm_remove_shard', { index: idx, model: modelId }))) return;
        try {
          var resp = await App.authFetch('/api/admin/models/' + encodeURIComponent(modelId) + '/shards/' + idx, { method: 'DELETE' });
          if (resp.ok) {
            App.ui.showBanner('success', 'Shard ' + idx + ' removed');
            App.models.load();
          } else {
            App.ui.showBanner('error', await U.getApiErrorMessage(resp, 'Failed to remove shard'));
          }
        } catch (e) {
          App.ui.showBanner('error', 'Remove failed: ' + e.message);
        }
      } else if (state === 'downloading') {
        App.models.cancelDownload(modelId);
      } else {
        // Single shard download — backend tries P2P first, falls back to HF
        try {
          var dlResp = await App.authFetch('/api/admin/models/' + encodeURIComponent(modelId) + '/shards/' + idx + '/download', { method: 'POST' });
          var dlData = await dlResp.json();
          if (dlData.status === 'downloading') {
            App.ui.showBanner('success', 'Downloading part ' + (idx + 1) + ' from ' + (dlData.source === 'p2p' ? 'peer ' + (dlData.peer || '') : 'peers'));
            App.models.load();
          } else if (dlData.status === 'use_hf') {
            // Backend says use HuggingFace
            var hfResp = await App.authFetch('/api/admin/hf/download-shards', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ repo_id: dlData.repo_id, filename: dlData.filename, shards: [idx], model_id: modelId }),
            });
            if (hfResp.ok) {
              App.ui.showBanner('success', 'Downloading part ' + (idx + 1) + ' from HuggingFace');
              App.models.load();
            } else {
              App.ui.showBanner('error', await U.getApiErrorMessage(hfResp, 'Download failed'));
            }
          } else if (dlData.status === 'already_local') {
            App.ui.showBanner('info', 'Part ' + (idx + 1) + ' is already on this device');
          } else {
            App.ui.showBanner('error', dlData.error ? dlData.error.message : 'Download unavailable');
          }
        } catch (e) {
          App.ui.showBanner('error', 'Download failed: ' + e.message);
        }
      }
    },

    toggleLock: async function() {
      var modelId = this.currentModel;
      var idx = this.currentIndex;
      var newLocked = !this.currentLocked;
      this.hide();
      try {
        var resp = await App.authFetch('/api/admin/models/' + encodeURIComponent(modelId) + '/shards/' + idx + '/lock', {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ locked: newLocked }),
        });
        if (resp.ok) {
          App.ui.showBanner('success', 'Shard ' + idx + (newLocked ? ' locked' : ' unlocked'));
          App.models.load();
        } else {
          App.ui.showBanner('error', 'Failed to update shard lock');
        }
      } catch (e) {
        App.ui.showBanner('error', 'Lock update failed: ' + e.message);
      }
    },

    loadShard: async function() {
      var modelId = this.currentModel;
      var idx = this.currentIndex;
      this.hide();

      try {
        var resp = await App.authFetch('/api/admin/models/' + encodeURIComponent(modelId) + '/shards/' + idx + '/load', { method: 'POST' });
        if (resp.ok) {
          App.notifications.showToast('Loading shard ' + (idx + 1) + ' into memory...', 'success');
          App.models.load();
        } else {
          App.notifications.showToast(await U.getApiErrorMessage(resp, 'Failed to load shard'), 'error');
        }
      } catch (e) {
        App.notifications.showToast('Load failed: ' + e.message, 'error');
      }
    },

    unloadShard: async function() {
      var modelId = this.currentModel;
      var idx = this.currentIndex;
      this.hide();

      if (!confirm('Unload shard ' + (idx + 1) + ' from memory?\n\nThe file stays on disk. The model worker will restart without this shard. Active inference may be briefly interrupted.')) return;

      try {
        // Unload this specific shard — narrows the shard window and restarts the worker.
        // The remaining shards stay loaded; only this one is freed.
        var resp = await App.authFetch('/api/admin/models/' + encodeURIComponent(modelId) + '/shards/' + idx + '/unload', { method: 'POST' });
        if (resp.ok) {
          var name = U.formatModelDisplayName(modelId);
          App.notifications.showToast('Shard ' + (idx + 1) + ' of ' + name + ' unloaded from memory', 'success');
          App.models.load();
        } else {
          App.notifications.showToast(await U.getApiErrorMessage(resp, 'Failed to unload'), 'error');
        }
      } catch (e) {
        App.notifications.showToast('Unload failed: ' + e.message, 'error');
      }
    }
  };
})();
