/**
 * Reference test models + diagnostics export.
 *
 * Reference models are pinned in the daemon (src/model/reference.rs) so that
 * speed numbers from different machines are comparable, and so a node with
 * nothing installed has something guaranteed to exist to fall back on rather
 * than waiting for the swarm to offer something.
 *
 * Nothing here downloads on its own. Every path is behind a button a person
 * pressed — testing the network is not a good enough reason to spend someone's
 * bandwidth without asking.
 */
(function () {
  if (!window.App) return;
  var U = App.utils;

  App.referenceModels = {
    _cache: null,

    /**
     * Fetch the pinned list. Cached — it only changes on daemon upgrade.
     *
     * Only a non-empty result is cached. An empty array is truthy in
     * JavaScript, so caching a failed fetch would satisfy the guard below
     * forever and disable the feature until the page was reloaded — a request
     * that happened to race startup auth would permanently cost the user the
     * cold-start offer. An empty result stays uncached so the next caller
     * retries.
     */
    load: function () {
      var c = App.referenceModels._cache;
      if (c && c.length) {
        return Promise.resolve(c);
      }
      return App
        .authFetch('/api/admin/reference-models')
        .then(function (r) { return r.json(); })
        .then(function (d) {
          var models = (d && d.models) || [];
          if (models.length) App.referenceModels._cache = models;
          return models;
        })
        .catch(function () { return []; });
    },

    /** True when `modelId` is one of the pinned models. Drives the badge. */
    isReference: function (modelId) {
      var list = App.referenceModels._cache;
      if (!list || !modelId) return false;
      for (var i = 0; i < list.length; i++) {
        if (list[i].model_id === modelId) return true;
      }
      return false;
    },

    /**
     * Start acquiring a tier.
     *
     * `fairShare` asks for a slice sized against how many peers are taking
     * part rather than the whole model — usually what you want, since the
     * point is for the swarm to serve it collectively. Whole-model is offered
     * for a node that wants to answer on its own.
     */
    acquire: function (tier, fairShare) {
      return App.referenceModels.load().then(function (list) {
        var m = null;
        for (var i = 0; i < list.length; i++) {
          if (list[i].tier === tier) { m = list[i]; break; }
        }
        if (!m) return;

        var body = { repo_id: m.repo_id, filename: m.filename };
        if (fairShare) {
          body.shards = [];
          body.peer_fair_share = true;
        } else {
          body.shards = [];
          for (var j = 0; j < m.shards; j++) body.shards.push(j);
        }

        return App.utils
          .apiAction('/api/admin/hf/download-shards', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
          })
          .then(function () {
            App.notifications.showToast(
              I18n.t('reference.started', { model: m.model_id }),
              'success'
            );
            App.referenceModels.render();
          })
          .catch(function (e) {
            App.notifications.showToast(
              I18n.t('reference.failed', { error: U.extractErrorMessage(e) }),
              'error'
            );
          });
      });
    },

    /** Render the opt-in cards into the settings panel. */
    render: function () {
      var host = document.getElementById('reference-models-list');
      if (!host) return;
      App.referenceModels.load().then(function (list) {
        host.innerHTML = '';
        list.forEach(function (m) {
          var row = document.createElement('div');
          row.className = 'reference-row';

          var info = document.createElement('div');
          info.className = 'reference-row-info';

          var name = document.createElement('div');
          name.className = 'reference-row-name';
          name.textContent = I18n.t('reference.tier_' + m.tier);
          info.appendChild(name);

          var meta = document.createElement('div');
          meta.className = 'text-muted text-2xs';
          meta.textContent = I18n.t('reference.tier_' + m.tier + '_desc') +
            ' · ' + U.formatSize(m.size_mb) +
            ' · ' + I18n.t('reference.parts', { count: m.shards });
          info.appendChild(meta);
          row.appendChild(info);

          if (m.held) {
            var have = document.createElement('span');
            have.className = 'badge badge-green';
            have.textContent = I18n.t('reference.installed');
            row.appendChild(have);
          } else {
            var share = document.createElement('button');
            share.type = 'button';
            share.className = 'btn btn-sm';
            share.textContent = I18n.t('reference.get_share');
            share.title = I18n.t('reference.get_share_hint');
            share.addEventListener('click', function () {
              App.referenceModels.acquire(m.tier, true);
            });
            row.appendChild(share);

            var all = document.createElement('button');
            all.type = 'button';
            all.className = 'btn btn-sm btn-ghost';
            all.textContent = I18n.t('reference.get_all');
            all.title = I18n.t('reference.get_all_hint');
            all.addEventListener('click', function () {
              App.referenceModels.acquire(m.tier, false);
            });
            row.appendChild(all);
          }

          host.appendChild(row);
        });
      });
    },

    /**
     * Copy a pasteable node summary.
     *
     * Served as plain text by the daemon, redacted there — no key, no invite
     * code, no file paths, and every network address replaced by a placeholder
     * naming only its kind. That last part is why the endpoint is called
     * without `?full=1`: the report carries this machine's addresses and up to
     * ten remembered peer multiaddrs, which on a live node are other people's
     * home IP addresses, and the person clicking this button is not expected
     * to read the text before pasting it somewhere public.
     *
     * Falls back to a selectable textarea because navigator.clipboard is
     * unavailable over plain HTTP on a LAN address, which is exactly how most
     * people reach this dashboard.
     */
    copyDiagnostics: function () {
      return App
        .authFetch('/api/admin/diagnostics')
        .then(function (r) { return r.text(); })
        .then(function (text) {
          if (navigator.clipboard && window.isSecureContext) {
            return navigator.clipboard.writeText(text).then(function () {
              App.notifications.showToast(I18n.t('reference.copied'), 'success');
            });
          }
          App.referenceModels._showFallback(text);
        })
        .catch(function (e) {
          App.notifications.showToast(
            I18n.t('reference.copy_failed', { error: U.extractErrorMessage(e) }),
            'error'
          );
        });
    },

    _showFallback: function (text) {
      var host = document.getElementById('reference-models-list');
      if (!host) return;
      var existing = document.getElementById('diagnostics-fallback');
      if (existing) existing.remove();

      var wrap = document.createElement('div');
      wrap.id = 'diagnostics-fallback';
      wrap.className = 'mt-2';
      var hint = document.createElement('div');
      hint.className = 'text-muted text-2xs mb-1';
      hint.textContent = I18n.t('reference.copy_manual');
      var ta = document.createElement('textarea');
      ta.className = 'mono text-2xs';
      ta.rows = 12;
      ta.style.width = '100%';
      ta.readOnly = true;
      ta.value = text;
      wrap.appendChild(hint);
      wrap.appendChild(ta);
      host.parentNode.insertBefore(wrap, host.nextSibling);
      ta.focus();
      ta.select();
    },

    init: function () {
      var btn = document.getElementById('copy-diagnostics-btn');
      if (btn) {
        btn.addEventListener('click', App.referenceModels.copyDiagnostics);
      }
      var section = document.getElementById('settings-testing-section');
      if (section) {
        // Render on first expand rather than on page load — the list is only
        // needed once someone looks at it.
        section.addEventListener('toggle', function () {
          if (section.open) App.referenceModels.render();
        });
      }
      // Warm the cache so isReference() can badge models on the first render
      // of the models list without a race.
      //
      // Guarded because init() runs inside init.js's startup sequence: an
      // exception here aborts everything after it, which is how a wrong
      // accessor in this file took the whole dashboard's translations down.
      // Nothing this component does is worth failing startup over.
      try {
        App.referenceModels.load();
      } catch (e) {
        /* optional surface — never block startup */
      }
    },
  };
})();
