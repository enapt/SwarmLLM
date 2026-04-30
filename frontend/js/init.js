'use strict';

// ============================================================================
// SwarmLLM — Init + Event Binding
// Binds all UI events, initializes subsystems, exports window.SwarmLLM
// ============================================================================

(function() {
  var S = App.state;
  var U = App.utils;

  // Single source of truth for supported UI languages.
  // Tuple: [lang_code, display_name, country_code_for_flag (uppercase)]
  var LANGS = [
    ['en','English','GB'],['es','Espa\u00f1ol','ES'],['fr','Fran\u00e7ais','FR'],['de','Deutsch','DE'],
    ['pt','Portugu\u00eas','BR'],['it','Italiano','IT'],['nl','Nederlands','NL'],['ru','\u0420\u0443\u0441\u0441\u043a\u0438\u0439','RU'],
    ['zh','\u4e2d\u6587','CN'],['ja','\u65e5\u672c\u8a9e','JP'],['ko','\ud55c\uad6d\uc5b4','KR'],['ar','\u0627\u0644\u0639\u0631\u0628\u064a\u0629','SA'],
    ['tr','T\u00fcrk\u00e7e','TR'],['pl','Polski','PL'],['sv','Svenska','SE'],['th','\u0e44\u0e17\u0e22','TH'],
    ['hi','\u0939\u093f\u0928\u094d\u0926\u0940','IN'],['vi','Ti\u1ebfng Vi\u1ec7t','VN'],['id','Bahasa Indonesia','ID'],
    ['uk','\u0423\u043a\u0440\u0430\u0457\u043d\u0441\u044c\u043a\u0430','UA'],['cs','\u010ce\u0161tina','CZ']
  ];
  function langFlagCode(lang) {
    var entry = LANGS.find(function(l) { return l[0] === lang; });
    return entry ? entry[2].toLowerCase() : null;
  }

  // ========================================================================
  // Bind all UI event listeners
  // ========================================================================
  function bindEvents() {
    function on(id, event, fn) {
      var el = document.getElementById(id);
      if (el) el.addEventListener(event, fn);
    }

    // Tab buttons
    document.querySelectorAll('.tab-btn[data-tab]').forEach(function(btn) {
      btn.addEventListener('click', function() { App.ui.switchTab(btn.dataset.tab); });
    });

    // Setup wizard
    on('btn-prev', 'click', function() { App.setup.prevStep(); });
    on('btn-next', 'click', function() { App.setup.nextStep(); });
    on('btn-skip-setup', 'click', function(e) {
      e.preventDefault();
      App.setup.finish();
    });
    on('setup-provider-save', 'click', function() { App.setup.saveAndTestProvider(); });
    on('setup-provider-paste', 'click', function() { App.setup.pasteFromClipboard(); });
    // Submit on Enter inside the key input
    on('setup-provider-key', 'keydown', function(e) {
      if (e.key === 'Enter') { e.preventDefault(); App.setup.saveAndTestProvider(); }
    });
    // "Finish setup" chip in the dashboard header
    on('setup-finish-chip-main', 'click', function() { App.setup.reopen(); });
    on('setup-finish-chip-dismiss', 'click', function(e) {
      e.stopPropagation();
      App.setup.dismissChip();
    });
    // Wizard step indicators
    document.querySelectorAll('.wizard-step[data-step]').forEach(function(stepBtn) {
      stepBtn.addEventListener('click', function() {
        var target = parseInt(stepBtn.getAttribute('data-step'), 10);
        if (target < App.setup.currentStep) {
          App.setup.currentStep = target;
          App.setup.updateUI();
        }
      });
    });

    // Settings modal
    on('btn-close-settings', 'click', function() { App.ui.closeSettings(); });
    on('btn-copy-api-key', 'click', function() { App.settings.copyApiKey(); });
    on('btn-save-settings', 'click', function() { App.settings.save(); });
    on('btn-open-settings', 'click', function() { App.ui.openSettings(); });

    // Theme toggle
    on('btn-theme-toggle', 'click', function() {
      var themes = ['dark', 'light', 'system'];
      var icons = { dark: '\u263E', light: '\u2600', system: '\u25D1' };
      var cur = localStorage.getItem(App.THEME_KEY) || 'dark';
      var next = themes[(themes.indexOf(cur) + 1) % themes.length];
      localStorage.setItem(App.THEME_KEY, next);
      App.applyTheme(next);
      var btn = document.getElementById('btn-theme-toggle');
      if (btn) btn.textContent = icons[next] || '\u263E';
    });

    // Language picker dropdown
    (function() {
      // Country flag as inline SVG img
      function countryFlag(cc) {
        return '<img src="/static/flags/' + cc.toLowerCase() + '.svg" alt="' + cc + '" class="lang-flag-img">';
      }
      var dropdown = document.getElementById('lang-dropdown');
      var btn = document.getElementById('btn-lang-picker');
      if (!dropdown || !btn) return;
      LANGS.forEach(function(pair) {
        var b = document.createElement('button');
        b.type = 'button';
        b.innerHTML = countryFlag(pair[2]) + ' ' + App.utils.escapeHtml(pair[1]);
        b.title = pair[1];
        b.dataset.lang = pair[0];
        b.addEventListener('click', function() {
          if (typeof I18n !== 'undefined') I18n.setLang(pair[0]);
          var settingsLang = document.getElementById('settings-language');
          if (settingsLang) settingsLang.value = pair[0];
          dropdown.style.display = 'none';
          updateLangDropdownActive();
        });
        dropdown.appendChild(b);
      });
      btn.addEventListener('click', function(e) {
        e.stopPropagation();
        var open = dropdown.style.display !== 'none';
        dropdown.style.display = open ? 'none' : '';
        if (!open) updateLangDropdownActive();
      });
      document.addEventListener('click', function() { dropdown.style.display = 'none'; });
      dropdown.addEventListener('click', function(e) { e.stopPropagation(); });

      function updateLangDropdownActive() {
        var cur = (typeof I18n !== 'undefined') ? I18n.getLang() : 'en';
        dropdown.querySelectorAll('button').forEach(function(b) {
          b.classList.toggle('active', b.dataset.lang === cur);
        });
        // Update header button to show current flag
        var curLang = LANGS.find(function(l) { return l[0] === cur; });
        if (curLang && btn) btn.innerHTML = countryFlag(curLang[2]);
      }
      updateLangDropdownActive();
    })();

    // Setup wizard language picker (button-style dropdown — matches the header lang picker UX)
    (function() {
      var btn = document.getElementById('setup-lang-btn');
      var dropdown = document.getElementById('setup-lang-dropdown');
      if (!btn || !dropdown) return;

      function countryFlag(cc) {
        return '<img src="/static/flags/' + cc.toLowerCase() + '.svg" alt="' + cc + '" class="lang-flag-img">';
      }
      function refreshButton() {
        var cur = (typeof I18n !== 'undefined') ? I18n.getLang() : 'en';
        var entry = LANGS.find(function(l) { return l[0] === cur; }) || LANGS[0];
        var flag = document.getElementById('setup-lang-flag');
        var name = document.getElementById('setup-lang-name');
        if (flag) flag.src = '/static/flags/' + entry[2].toLowerCase() + '.svg';
        if (name) name.textContent = entry[1];
        var engBtn = document.getElementById('setup-lang-english');
        if (engBtn) engBtn.classList.toggle('hidden', cur === 'en');
      }
      LANGS.forEach(function(pair) {
        var b = document.createElement('button');
        b.type = 'button';
        b.setAttribute('role', 'option');
        b.innerHTML = countryFlag(pair[2]) + ' ' + App.utils.escapeHtml(pair[1]);
        b.dataset.lang = pair[0];
        b.addEventListener('click', function() {
          if (typeof I18n !== 'undefined') I18n.setLang(pair[0]);
          var settingsLang = document.getElementById('settings-language');
          if (settingsLang) settingsLang.value = pair[0];
          dropdown.style.display = 'none';
          btn.setAttribute('aria-expanded', 'false');
          refreshButton();
          syncDropdownActive();
          // Re-populate the summary if we're on Step 3 — the previous translation is stale.
          if (App.setup && App.setup.currentStep === 3) App.setup.populateSummary();
        });
        dropdown.appendChild(b);
      });
      function syncDropdownActive() {
        var cur = (typeof I18n !== 'undefined') ? I18n.getLang() : 'en';
        dropdown.querySelectorAll('button').forEach(function(b) {
          b.classList.toggle('active', b.dataset.lang === cur);
        });
      }
      btn.addEventListener('click', function(e) {
        e.stopPropagation();
        var open = dropdown.style.display !== 'none';
        dropdown.style.display = open ? 'none' : '';
        btn.setAttribute('aria-expanded', open ? 'false' : 'true');
        if (!open) syncDropdownActive();
      });
      document.addEventListener('click', function() {
        dropdown.style.display = 'none';
        btn.setAttribute('aria-expanded', 'false');
      });
      dropdown.addEventListener('click', function(e) { e.stopPropagation(); });
      // "Continue in English" pill (only visible when current language ≠ en).
      var engBtn = document.getElementById('setup-lang-english');
      if (engBtn) {
        engBtn.addEventListener('click', function() {
          if (typeof I18n !== 'undefined') I18n.setLang('en');
          var settingsLang = document.getElementById('settings-language');
          if (settingsLang) settingsLang.value = 'en';
          refreshButton();
          syncDropdownActive();
          if (App.setup && App.setup.currentStep === 3) App.setup.populateSummary();
        });
      }
      refreshButton();
      syncDropdownActive();
    })();

    on('btn-rerun-setup', 'click', function() {
      App.ui.closeSettings();
      App.setup.reopen();
    });

    on('btn-show-all-peers', 'click', function() {
      App.dashboard._peersExpanded = !App.dashboard._peersExpanded;
      if (App.dashboard._lastPeers && App.dashboard._lastPeers.length > 0) {
        App.dashboard.renderPeers(App.dashboard._lastPeers);
      } else {
        App.dashboard.loadNetworkData();
      }
    });

    // Provider test buttons
    document.querySelectorAll('[data-test-provider]').forEach(function(btn) {
      btn.addEventListener('click', function() {
        App.settings.testProvider(btn.getAttribute('data-test-provider'));
      });
    });

    // Provider filter
    var providerFilter = document.getElementById('provider-filter');
    if (providerFilter) {
      providerFilter.addEventListener('input', function() {
        var q = this.value.toLowerCase();
        var cards = document.querySelectorAll('#provider-cards .provider-card');
        cards.forEach(function(card) {
          var name = (card.querySelector('strong') || {}).textContent || '';
          card.style.display = name.toLowerCase().indexOf(q) >= 0 ? '' : 'none';
        });
      });
    }

    // Key source dropdown
    var keySourceSel = document.getElementById('provider-key-source');
    if (keySourceSel) {
      keySourceSel.addEventListener('change', function() {
        App.authFetch('/api/admin/providers', {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ key_source: this.value })
        }).then(function() {
          App.ui.showBanner('success', I18n.t('init.key_source_updated', { source: keySourceSel.value }));
          App.settings.loadProviders();
        });
      });
    }

    // Model browser
    on('btn-close-model-browser', 'click', function() { App.ui.closeModelBrowser(); });
    on('btn-hf-search', 'click', function() { App.hf.search(); });
    on('hf-search-input', 'keydown', function(e) { if (e.key === 'Enter') App.hf.search(); });
    on('hf-sort', 'change', function() { App.hf.sortResults(); });
    // HF suggestion chips
    document.querySelectorAll('[data-hf-suggest]').forEach(function(chip) {
      chip.addEventListener('click', function() {
        var query = chip.getAttribute('data-hf-suggest');
        var input = document.getElementById('hf-search-input');
        if (input) { input.value = query; }
        var suggestions = document.getElementById('hf-suggestions');
        if (suggestions) suggestions.style.display = 'none';
        App.hf.search();
      });
    });
    on('btn-open-model-browser', 'click', function() { App.ui.openModelBrowser(); });
    on('btn-browse-hf', 'click', function() { App.ui.openModelBrowser(); });

    // Header
    on('hamburger-btn', 'click', function() { App.ui.toggleSidebar(); });
    on('logo', 'click', function() { App.ui.switchTab('dashboard'); });
    on('btn-shutdown', 'click', function() { App.models.shutdown(); });

    // Sidebar
    on('sidebar-overlay', 'click', function() { App.ui.closeSidebar(); });
    on('btn-new-session', 'click', function() { App.chat.newSession(); if (S.activeTab !== 'chat') App.ui.switchTab('chat'); });
    on('btn-close-sidebar', 'click', function() { App.ui.closeSidebar(); });

    // Float-mode sidebar hover
    var _sidebarHoverTimer = null;
    var sidebarEl = document.getElementById('sidebar');
    if (sidebarEl) {
      sidebarEl.addEventListener('mouseenter', function() {
        clearTimeout(_sidebarHoverTimer);
        if (this.classList.contains('sidebar-float')) this.classList.remove('collapsed');
      });
      sidebarEl.addEventListener('mouseleave', function() {
        if (this.classList.contains('sidebar-float')) {
          _sidebarHoverTimer = setTimeout(function() {
            var s = document.getElementById('sidebar');
            if (s && s.classList.contains('sidebar-float')) s.classList.add('collapsed');
          }, 120);
        }
      });
    }

    // Chat
    on('send-btn', 'click', function() { App.chat.send(); });
    on('chat-input', 'keydown', function(e) { App.chat.handleKey(e); });
    // Image upload
    on('image-upload-btn', 'click', function() {
      document.getElementById('image-upload-input').click();
    });
    on('image-upload-input', 'change', function(e) {
      Array.from(e.target.files).forEach(function(f) { App.chat.addPendingImage(f); });
      e.target.value = '';
    });

    // Image paste
    var chatInput = document.getElementById('chat-input');
    if (chatInput) {
      chatInput.addEventListener('paste', function(e) {
        var items = (e.clipboardData || {}).items || [];
        for (var i = 0; i < items.length; i++) {
          if (items[i].type.indexOf('image') !== -1) {
            e.preventDefault();
            App.chat.addPendingImage(items[i].getAsFile());
          }
        }
      });
    }

    // Image drag-and-drop
    var chatArea = document.getElementById('view-chat');
    if (chatArea) {
      chatArea.addEventListener('dragover', function(e) {
        e.preventDefault();
        e.dataTransfer.dropEffect = 'copy';
      });
      chatArea.addEventListener('drop', function(e) {
        e.preventDefault();
        Array.from(e.dataTransfer.files).forEach(function(f) {
          if (f.type.startsWith('image/')) App.chat.addPendingImage(f);
        });
      });
    }

    // Delegated CTA buttons
    document.addEventListener('click', function(e) {
      var el = e.target.closest('[data-goto-chat],[data-goto-browse],[data-goto-settings],[data-goto-hf],[data-goto-network-code]') || e.target;
      if (el.getAttribute('data-goto-chat')) { App.ui.switchTab('chat'); }
      if (el.getAttribute('data-goto-browse')) { App.ui.openModelBrowser(); }
      if (el.getAttribute('data-goto-settings')) { App.ui.openSettings(true); }
      if (el.getAttribute('data-goto-hf')) { App.ui.openModelBrowser(); }
      if (el.getAttribute('data-goto-network-code')) { var btn = document.getElementById('btn-share-network'); if (btn) btn.click(); }
    });

    // Network discovery — share popover
    on('btn-share-network', 'click', function(e) {
      e.stopPropagation();
      var pop = document.getElementById('share-popover');
      if (pop) pop.classList.toggle('show');
    });
    on('btn-copy-network-code', 'click', function() { App.networkCode.copy(); });
    on('btn-join-network', 'click', function() { App.networkCode.join(); });

    // Network map
    on('map-model-filter', 'change', function() { App.networkMap.applyFilter(); });
    on('btn-refresh-map', 'click', function() { App.networkMap.refresh(); });

    // Model Compare
    on('btn-compare-run', 'click', function() { App.compare.run(); });

    // Leaderboard
    on('btn-refresh-leaderboard', 'click', function() { App.identity.loadLeaderboard(); });

    // Escape key + Tab focus trap
    document.addEventListener('keydown', function(e) {
      if (e.key === 'Escape') {
        var sidebar = document.getElementById('sidebar');
        var settingsModal = document.getElementById('settings-modal');
        var modelModal = document.getElementById('model-browser-modal');
        if (sidebar && !sidebar.classList.contains('collapsed') && window.innerWidth < 768) { App.ui.closeSidebar(); }
        else if (settingsModal && !settingsModal.classList.contains('hidden')) { App.ui.closeSettings(); }
        else if (modelModal && !modelModal.classList.contains('hidden')) { App.ui.closeModelBrowser(); }
      }
      if (e.key === 'Tab') {
        var openModal = document.querySelector('.modal-overlay:not(.hidden) .modal');
        if (openModal) {
          var focusable = openModal.querySelectorAll('button, [href], input:not([type="hidden"]), select, textarea, [tabindex]:not([tabindex="-1"])');
          if (focusable.length > 0) {
            var first = focusable[0], last = focusable[focusable.length - 1];
            if (e.shiftKey && document.activeElement === first) { e.preventDefault(); last.focus(); }
            else if (!e.shiftKey && document.activeElement === last) { e.preventDefault(); first.focus(); }
          }
        }
      }
    });

    // Double-click to rename session
    document.addEventListener('dblclick', function(e) {
      var target = e.target;
      var renameId = target.getAttribute('data-rename-session');
      if (renameId) {
        e.stopPropagation();
        e.preventDefault();
        App.chat.renameSession(renameId, target);
      }
    });

    // Delegated click handlers for dynamic elements
    document.addEventListener('click', function(e) {
      var target = e.target;

      // Close share popover
      var pop = document.getElementById('share-popover');
      if (pop && pop.classList.contains('show')) {
        var wrap = document.querySelector('.share-btn-wrap');
        if (wrap && !wrap.contains(target)) pop.classList.remove('show');
      }

      // Session delete
      var delId = target.getAttribute('data-delete-session');
      if (delId) { e.stopPropagation(); App.chat.deleteSession(delId, e); return; }

      // Chat header title rename
      if (target.id === 'chat-header-title' && S.currentSessionId) {
        App.chat.renameSession(S.currentSessionId, target);
        return;
      }

      // Model action buttons
      var selectId = target.getAttribute('data-select-model');
      if (selectId) { App.models.select(selectId); return; }

      var cloudRow = target.closest('[data-select-cloud]');
      if (cloudRow) { App.models.selectDropdown(cloudRow.getAttribute('data-select-cloud')); App.chat.newSession(); App.ui.switchTab('chat'); return; }

      var cancelId = target.getAttribute('data-cancel-download');
      if (cancelId) { App.models.cancelDownload(cancelId); return; }

      var requestId = target.getAttribute('data-request-model');
      if (requestId) { App.models.request(requestId); return; }

      var removeId = target.getAttribute('data-remove-model');
      if (removeId) { App.models.remove(removeId); return; }

      var unloadId = target.getAttribute('data-unload-model');
      if (unloadId) { App.models.unload(unloadId); return; }

      // HF download
      var hfRepo = target.getAttribute('data-hf-download');
      if (hfRepo) { App.hf.download(hfRepo, target.getAttribute('data-hf-variant') || ''); return; }

      // Per-model activity ticker: click to expand history
      var tickerEl = target.closest('.model-ticker');
      if (tickerEl && !target.closest('button, a')) {
        tickerEl.classList.toggle('expanded');
        e.stopPropagation();
        return;
      }

      // Shard row inline actions (load/unload/download/cancel/delete/toggle-lock/expand)
      var shardAct = target.getAttribute('data-shard-act');
      if (shardAct) {
        var rowEl = target.closest('.shard-row');
        if (rowEl) {
          var modelId = rowEl.getAttribute('data-shard-model');
          var idx = parseInt(rowEl.getAttribute('data-shard-index'), 10);
          if (modelId != null && !isNaN(idx)) {
            App.dashboard.shardRowAction(shardAct, modelId, idx, rowEl);
            e.stopPropagation();
            return;
          }
        }
      }

      // Shard view toggle (List | Matrix)
      var shardView = target.getAttribute('data-shard-view');
      if (shardView && target.closest('.shard-view-toggle')) {
        App.dashboard.setShardView(shardView);
        e.stopPropagation();
        return;
      }

      // Matrix "show all peers" button
      var matrixShowAll = target.getAttribute('data-matrix-showall');
      if (matrixShowAll) {
        App.dashboard.expandMatrixAllPeers(matrixShowAll);
        e.stopPropagation();
        return;
      }

      // GGUF metadata toggle
      var metaToggle = target.getAttribute('data-meta-toggle');
      if (metaToggle) { App.models.toggleMetadata(metaToggle); return; }

      // Download queue cancel
      var dlCancel = target.getAttribute('data-dl-cancel');
      if (dlCancel) { App.downloads.cancelDownload(dlCancel); return; }

      // Download queue log toggle
      var dlLogToggle = target.getAttribute('data-dl-log-toggle');
      if (dlLogToggle) {
        var logEl = document.querySelector('[data-dl-log="' + dlLogToggle + '"]');
        if (logEl) logEl.classList.toggle('open');
        return;
      }

      // Encrypted pipeline toggle
      var encToggle = target.getAttribute('data-enc-toggle') || (target.closest('[data-enc-toggle]') || {}).getAttribute && (target.closest('[data-enc-toggle]') || {}).getAttribute('data-enc-toggle');
      if (encToggle) {
        var encReady = (target.getAttribute('data-enc-ready') || (target.closest('[data-enc-ready]') || {}).getAttribute && (target.closest('[data-enc-ready]') || {}).getAttribute('data-enc-ready')) === '1';
        if (encReady) {
          var encModelData = (App.data.cache.models || []).find(function(m) { return m.id === encToggle; });
          var isActive = encModelData ? !!encModelData.encrypted_pipeline : (target.classList.contains('active') || target.closest('.active') != null);
          App.authFetch(U.modelApiUrl(encToggle, 'encrypted-pipeline'), {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ enabled: !isActive }),
          }).then(function(r) {
            if (r.ok) {
              var key = isActive ? 'enc.pipeline_disabled' : 'enc.pipeline_enabled';
              App.ui.showBanner('success', I18n.t(key, { model: encToggle }));
              App.models.load();
            } else {
              App.ui.showBanner('error', I18n.t('errors.failed_toggle_enc'));
            }
          });
        } else {
          if (confirm(I18n.t('init.confirm_download_shards'))) {
            App.authFetch('/api/admin/hf/source/' + encodeURIComponent(encToggle)).then(function(r) {
              if (!r.ok) { App.ui.showBanner('error', I18n.t('init.no_hf_source', { model: encToggle })); return; }
              return r.json();
            }).then(function(src) {
              if (!src) return;
              var modelData = (App.data.cache.models || []).find(function(mm) { return mm.id === encToggle; });
              var missing = [];
              if (modelData) {
                var first = modelData.shards[0];
                var last = modelData.shards[modelData.shards.length - 1];
                if (first && !first.local) missing.push(first.index);
                if (last && !last.local) missing.push(last.index);
              }
              if (missing.length === 0) { App.ui.showBanner('info', I18n.t('init.no_missing_shards')); return; }
              App.hf.downloadShards({ repo_id: src.repo_id, filename: src.filename, shards: missing }).then(function(result) {
                if (result.ok) App.ui.showBanner('success', I18n.t('init.downloading_shards', { shards: missing.join(', ') }));
                else App.ui.showBanner('error', result.errorMsg);
              });
            });
          }
        }
        return;
      }

      // Auto-manage gear icon
      var gearId = target.getAttribute('data-am-gear');
      if (gearId) { App.models.toggleAutoManage(gearId); return; }

      // Auto-manage save
      var amSave = target.getAttribute('data-am-save');
      if (amSave) { App.models.saveAutoManage(amSave); return; }

      // Model card expand/collapse toggle — chevron or title row click
      var expandBtn = target.closest('[data-expand-model]');
      var titleRow = !expandBtn && target.closest('.model-card-title');
      if (expandBtn || (titleRow && !target.closest('button, select, input, .badge-encrypted, [data-am-gear], [data-meta-toggle]'))) {
        var expandCard = (expandBtn || titleRow).closest('.model-card');
        if (expandCard && expandCard.getAttribute('data-model-id')) {
          var expandModelId = expandCard.getAttribute('data-model-id');
          var wasCompact = expandCard.classList.contains('compact');
          if (wasCompact) {
            expandCard.classList.remove('compact');
            App.state._expandedModels[expandModelId] = true;
            if (App.dashboard && App.dashboard._measurePipelineConnector) {
              requestAnimationFrame(function() { App.dashboard._measurePipelineConnector(expandCard); });
            }
          } else {
            expandCard.classList.add('compact');
            delete App.state._expandedModels[expandModelId];
          }
        }
        return;
      }

      // Peer table sort click
      var peerSortTh = target.closest('[data-peer-sort]');
      if (peerSortTh) {
        var key = peerSortTh.getAttribute('data-peer-sort');
        if (App.dashboard._peerSort === key) {
          App.dashboard._peerSortDir = App.dashboard._peerSortDir === 'asc' ? 'desc' : 'asc';
        } else {
          App.dashboard._peerSort = key;
          App.dashboard._peerSortDir = key === 'name' ? 'asc' : 'desc';
        }
        App.dashboard.renderPeers(App.dashboard._lastPeers);
        return;
      }

      // Cloud provider expand/collapse toggle — click on Browse button OR card header
      var cloudExpand = target.closest('[data-cloud-expand]');
      var cloudHeader = !cloudExpand && target.closest('.cloud-card-header');
      if (cloudExpand || cloudHeader) {
        var cloudCard = (cloudExpand || cloudHeader).closest('.model-card.cloud-model');
        if (cloudCard) {
          cloudCard.classList.toggle('cloud-card-collapsed');
          var isCollapsed = cloudCard.classList.contains('cloud-card-collapsed');
          var toggleEl = cloudCard.querySelector('[data-cloud-expand]');
          if (toggleEl) toggleEl.innerHTML = isCollapsed ? '&#9662;' : '&#9652;';
        }
        return;
      }

      // Availability bar click -> expand card
      var availBar = target.closest('.availability-bar');
      if (availBar) {
        var availCard = availBar.closest('.model-card');
        if (availCard && availCard.classList.contains('compact')) {
          var availModelId = availCard.getAttribute('data-model-id');
          if (availModelId) {
            availCard.classList.remove('compact');
            App.state._expandedModels[availModelId] = true;
          }
        }
        return;
      }

      // Model card click -> select and chat
      var modelCard = target.closest('.model-card');
      if (modelCard && !target.closest('button, a, summary, details, .shard-row, .badge-encrypted, [data-cancel-download], [data-remove-model], [data-unload-model], [data-enc-toggle], [data-am-gear], input, select, .model-expand-chevron, .availability-bar')) {
        var cardModelId = modelCard.getAttribute('data-model-id');
        if (cardModelId) {
          var cardModel = (App.data.cache.models || []).find(function(mm) { return mm.id === cardModelId; });
          var cardReady = cardModel && (cardModel.status === 'loaded' || cardModel.status === 'ready' ||
            (cardModel.global_available === cardModel.shard_count && cardModel.shard_count > 0));
          if (cardReady) {
            App.models.selectDropdown(cardModelId);
            App.chat.newSession();
            App.ui.switchTab('chat');
          } else {
            App.ui.showBanner('warning', I18n.t('init.model_not_ready'));
          }
          return;
        }
      }

      // Compare card copy
      var copyCompare = target.getAttribute('data-copy-compare');
      if (copyCompare) {
        var el = document.getElementById(copyCompare);
        if (el) {
          U.copyToClipboard(el.textContent, {
            btn: target,
            successLabel: I18n.t('actions.copied'),
            resetLabel: I18n.t('actions.copy'),
            duration: 1500,
          });
        }
        return;
      }

      // Compare history restore
      var historyRow = target.closest('[data-compare-idx]');
      if (historyRow) {
        var idx = parseInt(historyRow.getAttribute('data-compare-idx'), 10);
        try {
          var hist = JSON.parse(localStorage.getItem(App.COMPARE_HISTORY_KEY) || '[]');
          if (hist[idx]) App.compare.restoreFromHistory(hist[idx]);
        } catch (e) {}
        return;
      }

      // Chat action buttons (copy, compare)
      if (target.getAttribute('data-action') === 'copy') {
        var msgEl = target.closest('.chat-msg');
        var contentEl = msgEl ? msgEl.querySelector('.msg-content') : null;
        if (contentEl) {
          U.copyToClipboard(contentEl.textContent, {
            btn: target,
            successLabel: I18n.t('actions.copied'),
            resetLabel: I18n.t('actions.copy'),
            duration: 1500,
          });
        }
        return;
      }
      if (target.getAttribute('data-action') === 'compare') {
        var msgEl2 = target.closest('.chat-msg');
        if (msgEl2) {
          var prev = msgEl2.previousElementSibling;
          while (prev && !prev.classList.contains('user')) prev = prev.previousElementSibling;
          if (prev) {
            var userContent = prev.querySelector('.msg-content');
            if (userContent) {
              App.ui.switchTab('compare');
              var promptEl = document.getElementById('compare-prompt');
              if (promptEl) promptEl.value = userContent.textContent;
              App.compare.loadModels();
              App.notifications.showToast(I18n.t('init.compare_ready'), 'info');
            }
          }
        }
        return;
      }

      // Chat retry
      if (target.getAttribute('data-retry-chat')) {
        var errMsg = target.closest('.chat-msg');
        if (errMsg) errMsg.remove();
        if (S.currentSessionId && S.sessions[S.currentSessionId]) {
          var msgs = S.sessions[S.currentSessionId].messages;
          if (msgs.length > 0 && msgs[msgs.length - 1].role === 'user') {
            var lastUserMsg = msgs.pop();
            App.chat.saveSessions();
            document.getElementById('chat-input').value = lastUserMsg.content;
            App.chat.send();
          }
        }
        return;
      }

    });
  }

  // ========================================================================
  // Collapsible Panels
  // ========================================================================
  function initCollapsiblePanels() {
    document.querySelectorAll('.panel-header[data-collapse]').forEach(function(header) {
      header.addEventListener('click', function() {
        var targetId = header.getAttribute('data-collapse');
        var body = document.getElementById(targetId);
        if (!body) return;
        body.classList.toggle('collapsed');
        header.classList.toggle('collapsed');
      });
    });
  }

  // Populate .provider-signup-link[href] for each .provider-card[data-provider]
  // from PROVIDER_SIGNUP_URLS (providers.js). Keeps one source of truth for
  // provider signup URLs — previously duplicated in init.js + static HTML.
  function populateProviderSignupLinks() {
    if (typeof PROVIDER_SIGNUP_URLS === 'undefined') return;
    var cards = document.querySelectorAll('.provider-card[data-provider]');
    for (var i = 0; i < cards.length; i++) {
      var key = cards[i].getAttribute('data-provider');
      var url = PROVIDER_SIGNUP_URLS[key];
      if (!url) continue;
      var link = cards[i].querySelector('.provider-signup-link');
      if (link) link.href = url;
    }
  }

  // ========================================================================
  // Init
  // ========================================================================
  function initAfterI18n() {
    bindEvents();
    initCollapsiblePanels();
    populateProviderSignupLinks();
    App.models.initDropdown();
    App.models.initMobileSync();

    S.inputEl = document.getElementById('chat-input');
    if (S.inputEl) {
      S.inputEl.addEventListener('input', U.autoResizeInput);
      S.inputEl.addEventListener('input', U.updateTokenCounter);
    }

    App.setup.init();
    App.settings.init();
    if (App.pool) App.pool.init();
    if (App.autoManageStatus) App.autoManageStatus.init();
    if (App.claudeCode) App.claudeCode.init();
    if (App.responses) App.responses.init();
    App.settings._apiKeyPromise = App.settings.loadApiKey();

    App.ui.switchTab(S.activeTab, true);

    // Post-switch: load compare data if that tab is active
    if (S.activeTab === 'compare' && App.compare) {
      setTimeout(function() {
        App.compare.loadModels();
        App.compare.renderHistory();
      }, 0);
    }

    // Data loading — must happen after i18n + UI init
    App.chat.loadSessions();
    App.chat.renderSessionList();
    App.chat.renderMessages();

    App.applyTheme(localStorage.getItem(App.THEME_KEY) || 'dark');

    if (typeof NeuralBg !== 'undefined') NeuralBg.init();

    App.dashboard.loadInitial();
    App.pruneSchedule.loadHistory();
    App.pruneSchedule.loadSchedule();
    App.modeIndicator.load();
    App.identity.loadNickname();
    App.notifications.connectWebSocket();
    App.notifications.startPolling();
    App.providerHealth.startHealthPolling();

    window.addEventListener('popstate', function(e) {
      var tab = (e.state && e.state.tab) ? e.state.tab : 'dashboard';
      App.ui.switchTab(tab, true);
    });
  }

  function init() {
    if (typeof I18n !== 'undefined') {
      I18n.init(['en','es','fr','de','pt','it','nl','ru','zh','ja','ko','ar','tr','pl','sv','th','hi','vi','id','uk','cs'], initAfterI18n);
      return;
    }
    initAfterI18n();
  }

  // Delegated error handler for provider icons
  document.addEventListener('error', function(e) {
    var t = e.target;
    if (t.tagName !== 'IMG' || !t.classList.contains('provider-icon')) return;
    if (t.classList.contains('provider-avatar-icon')) {
      var av = t.parentNode;
      if (av) av.textContent = (typeof I18n !== 'undefined') ? I18n.t('chat.avatar_ai') : 'AI';
    } else {
      t.style.display = 'none';
    }
  }, true);

  // Start when DOM is ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

  // Export public API
  window.SwarmLLM = {
    ui: App.ui,
    chat: App.chat,
    dashboard: App.dashboard,
    hf: App.hf,
    setup: App.setup,
    identity: App.identity,
    networkMap: App.networkMap,
    compare: App.compare,
    requestModel: function(id) { App.models.request(id); },
    selectModel: function(id) { App.models.select(id); },
    cancelDownload: function(id) { App.models.cancelDownload(id); },
    removeModel: function(id) { App.models.remove(id); },
    shutdown: function() { App.models.shutdown(); },
    copyNetworkCode: function() { App.networkCode.copy(); },
    joinNetwork: function() { App.networkCode.join(); },
    openModelBrowser: function() { App.ui.openModelBrowser(); },
    switchTab: function(t) { App.ui.switchTab(t); },
  };
})();
