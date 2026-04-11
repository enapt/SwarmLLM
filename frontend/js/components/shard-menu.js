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
        loadBtn.title = I18n.t('shard.load_tip');
      }

      // Unload button — only when shard is loaded in memory
      if (unloadBtn) {
        unloadBtn.style.display = (shardState === 'local' && isInVram) ? '' : 'none';
        unloadBtn.title = I18n.t('shard.unload_tip');
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
          warnEl.textContent = I18n.t('shard.auto_manage_warn');
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
        if (!confirm(I18n.t('actions.confirm_remove_shard', { index: idx + 1, model: modelId }))) return;
        try {
          var resp = await App.authFetch(U.modelApiUrl(modelId, 'shards', idx), { method: 'DELETE' });
          if (resp.ok) {
            App.ui.showBanner('success', I18n.t('shard.removed', { idx: idx + 1 }));
            App.models.load();
          } else {
            App.ui.showBanner('error', await U.getApiErrorMessage(resp, I18n.t('shard.remove_failed')));
          }
        } catch (e) {
          App.ui.showBanner('error', I18n.t('shard.remove_error', { error: e.message }));
        }
      } else if (state === 'downloading') {
        App.models.cancelDownload(modelId);
      } else {
        // Single shard download — backend tries P2P first, falls back to HF
        try {
          var dlResp = await App.authFetch(U.modelApiUrl(modelId, 'shards', idx) + '/download', { method: 'POST' });
          var dlData = await dlResp.json();
          if (dlData.status === 'downloading') {
            App.ui.showBanner('success', I18n.t('shard.downloading_from', { idx: idx + 1, source: dlData.source === 'p2p' ? I18n.t('shard.source_peer', { id: dlData.peer || '' }) : I18n.t('shard.source_peers') }));
            App.models.load();
          } else if (dlData.status === 'use_hf') {
            // Backend says use HuggingFace
            var hfResult = await App.hf.downloadShards({ repo_id: dlData.repo_id, filename: dlData.filename, shards: [idx], model_id: modelId });
            if (hfResult.ok) {
              App.ui.showBanner('success', I18n.t('shard.downloading_hf', { idx: idx + 1 }));
              App.models.load();
            } else {
              App.ui.showBanner('error', hfResult.errorMsg || I18n.t('shard.hf_download_failed'));
            }
          } else if (dlData.status === 'already_local') {
            App.ui.showBanner('info', I18n.t('shard.already_local', { idx: idx + 1 }));
          } else {
            App.ui.showBanner('error', U.extractErrorMessage(dlData, I18n.t('shard.download_unavailable')));
          }
        } catch (e) {
          App.ui.showBanner('error', I18n.t('shard.download_failed', { error: e.message }));
        }
      }
    },

    toggleLock: async function() {
      var modelId = this.currentModel;
      var idx = this.currentIndex;
      var newLocked = !this.currentLocked;
      this.hide();
      var url = U.modelApiUrl(modelId, 'shards', idx) + '/lock';
      var opts = { method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ locked: newLocked }) };
      try {
        var resp = await App.authFetch(url, opts);
        if (resp.ok) {
          App.ui.showBanner('success', I18n.t(newLocked ? 'shard.locked' : 'shard.unlocked', { idx: idx + 1 }));
          App.models.load();
        } else {
          App.ui.showBanner('error', I18n.t('shard.lock_failed'));
        }
      } catch (e) {
        App.ui.showBanner('error', I18n.t('shard.lock_error', { error: e.message }));
      }
    },

    _shardAction: async function(url, opts, successMsg, failedKey, errorKey) {
      try {
        var resp = await App.authFetch(url, opts);
        if (resp.ok) {
          App.notifications.showToast(successMsg, 'success');
          App.models.load();
        } else {
          App.notifications.showToast(await U.getApiErrorMessage(resp, I18n.t(failedKey)), 'error');
        }
      } catch (e) {
        App.notifications.showToast(I18n.t(errorKey, { error: e.message }), 'error');
      }
    },

    loadShard: async function() {
      var modelId = this.currentModel;
      var idx = this.currentIndex;
      this.hide();
      var url = U.modelApiUrl(modelId, 'shards', idx) + '/load';
      await this._shardAction(url, { method: 'POST' }, I18n.t('shard.loading', { idx: idx + 1 }), 'shard.load_failed', 'shard.load_error');
    },

    unloadShard: async function() {
      var modelId = this.currentModel;
      var idx = this.currentIndex;
      this.hide();
      if (!confirm(I18n.t('shard.confirm_unload', { idx: idx + 1 }))) return;
      var url = U.modelApiUrl(modelId, 'shards', idx) + '/unload';
      var name = U.formatModelDisplayName(modelId);
      await this._shardAction(url, { method: 'POST' }, I18n.t('shard.unloaded', { idx: idx + 1, model: name }), 'shard.unload_failed', 'shard.unload_error');
    }
  };
})();
