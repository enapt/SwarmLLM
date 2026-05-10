'use strict';

// =============================================================================
// SwarmLLM — "The Swarm" tab (R111)
// Wishlist + Running-now + Search-HF + Capacity-plan sub-views. The wishlist
// is the user-visible face of auto-manage: instead of the daemon downloading
// models in mysterious silence, the user sees a ranked queue with status
// badges, plain-language "why" tags, and CTAs that explain what their
// contribution unlocks. Designed for non-technical users — every label is
// human-readable; no internal jargon (shards, replicas, layer ranges) leaks.
// =============================================================================

(function () {
  if (!window.App) return;

  var STATUS_LABELS = {
    hosting: 'wishlist.status.hosting',
    serveable: 'wishlist.status.serveable',
    aspirational: 'wishlist.status.aspirational',
    unreachable: 'wishlist.status.unreachable',
    blocked: 'wishlist.status.blocked',
  };

  // CSS class per status — drives the heat colour. Maps to the
  // .wishlist-pill-* rules added in style.css.
  var STATUS_CLASSES = {
    hosting: 'wishlist-pill-hosting',
    serveable: 'wishlist-pill-serveable',
    aspirational: 'wishlist-pill-aspirational',
    unreachable: 'wishlist-pill-unreachable',
    blocked: 'wishlist-pill-blocked',
  };

  // Tag-format helper — backend may emit `wishlist.why.parts_missing|missing=4`
  // (a key plus a `|var=val&var2=val2` payload). Decode into i18n params.
  function _parseTag(tag) {
    var pipe = tag.indexOf('|');
    if (pipe < 0) return { key: tag, params: {} };
    var key = tag.substring(0, pipe);
    var params = {};
    tag.substring(pipe + 1).split('&').forEach(function (pair) {
      var eq = pair.indexOf('=');
      if (eq < 0) return;
      params[pair.substring(0, eq)] = pair.substring(eq + 1);
    });
    return { key: key, params: params };
  }

  function _humaniseSize(mb) {
    if (mb < 1) return '< 1 MB';
    if (mb < 1024) return Math.round(mb) + ' MB';
    if (mb < 1024 * 1024) return (mb / 1024).toFixed(1) + ' GB';
    return (mb / (1024 * 1024)).toFixed(2) + ' TB';
  }

  // Build a single wishlist card. Returns a DocumentFragment.
  function _renderEntry(entry) {
    var tmpl = document.createElement('div');
    tmpl.className = 'wishlist-card';
    var statusClass = STATUS_CLASSES[entry.status] || 'wishlist-pill-blocked';
    tmpl.classList.add('wishlist-card-' + entry.status);

    // Header row: name + status pill + score heat
    var header = document.createElement('div');
    header.className = 'wishlist-card-header';
    var name = document.createElement('div');
    name.className = 'wishlist-card-name';
    name.textContent = entry.display_name || entry.model_id;
    header.appendChild(name);

    var statusPill = document.createElement('span');
    statusPill.className = 'wishlist-pill ' + statusClass;
    statusPill.textContent = I18n.t(STATUS_LABELS[entry.status] || STATUS_LABELS.blocked);
    if (entry.hosted_by_us) {
      statusPill.title = I18n.t('capacity.you_host_this');
    }
    header.appendChild(statusPill);

    var heat = document.createElement('span');
    heat.className = 'wishlist-heat';
    heat.title = I18n.t('wishlist.score_tip', { score: entry.score });
    var heatBar = document.createElement('span');
    heatBar.className = 'wishlist-heat-bar';
    heatBar.style.width = Math.max(2, Math.min(100, entry.score)) + '%';
    heat.appendChild(heatBar);
    header.appendChild(heat);
    tmpl.appendChild(header);

    // Meta row: size + memory required + replication summary
    var meta = document.createElement('div');
    meta.className = 'wishlist-card-meta';
    meta.appendChild(_metaSpan(I18n.t('wishlist.meta_size', { size: _humaniseSize(entry.size_mb) })));
    if (entry.vram_required_mb > 0) {
      meta.appendChild(_metaSpan(I18n.t('wishlist.meta_memory', { size: _humaniseSize(entry.vram_required_mb) })));
    }
    var replicaText;
    if (entry.swarm_replicas === 0) {
      replicaText = I18n.t('wishlist.meta_no_hosts');
    } else if (entry.target_replicas > 0 && entry.swarm_replicas < entry.target_replicas) {
      replicaText = I18n.t('wishlist.meta_replicas_partial', {
        have: entry.swarm_replicas,
        want: entry.target_replicas,
      });
    } else {
      replicaText = I18n.t('wishlist.meta_replicas_ok', { have: entry.swarm_replicas });
    }
    meta.appendChild(_metaSpan(replicaText));

    if (entry.total_shards > 0 && entry.shards_covered < entry.total_shards) {
      var pct = Math.floor((entry.shards_covered / entry.total_shards) * 100);
      meta.appendChild(_metaSpan(I18n.t('wishlist.meta_coverage', {
        covered: entry.shards_covered,
        total: entry.total_shards,
        pct: pct,
      })));
    }
    tmpl.appendChild(meta);

    // Why-tags row — each tag is an i18n key (or key|var=val payload).
    if (Array.isArray(entry.why_tags) && entry.why_tags.length > 0) {
      var tags = document.createElement('div');
      tags.className = 'wishlist-tags';
      entry.why_tags.forEach(function (raw) {
        var parsed = _parseTag(raw);
        var span = document.createElement('span');
        span.className = 'wishlist-tag';
        // Fallback to raw key if i18n missing — better than blank.
        var label = I18n.t(parsed.key, parsed.params);
        if (label === parsed.key) {
          // Last-ditch fallback so the user still sees something
          // semi-meaningful even when an i18n entry is missing.
          label = parsed.key.split('.').pop().replace(/_/g, ' ');
        }
        span.textContent = label;
        tags.appendChild(span);
      });
      tmpl.appendChild(tags);
    }

    // CTA row — the single most important action this card affords.
    var cta = document.createElement('div');
    cta.className = 'wishlist-cta';
    if (entry.status === 'hosting') {
      cta.appendChild(_actionLabel(I18n.t('wishlist.cta_hosting')));
    } else if (entry.status === 'serveable') {
      var btn = document.createElement('button');
      btn.className = 'btn btn-sm';
      btn.textContent = I18n.t('wishlist.cta_help_host');
      btn.dataset.modelId = entry.model_id;
      btn.addEventListener('click', function () { _onHelpHost(entry); });
      cta.appendChild(btn);
    } else if (entry.status === 'aspirational') {
      var btn2 = document.createElement('button');
      btn2.className = 'btn btn-sm btn-primary';
      btn2.textContent = I18n.t('wishlist.cta_aspirational');
      btn2.dataset.modelId = entry.model_id;
      btn2.addEventListener('click', function () { _onHelpHost(entry); });
      cta.appendChild(btn2);
    } else if (entry.status === 'unreachable') {
      cta.appendChild(_actionLabel(I18n.t('wishlist.cta_unreachable')));
    } else {
      cta.appendChild(_actionLabel(I18n.t('wishlist.cta_blocked')));
    }
    tmpl.appendChild(cta);

    return tmpl;
  }

  function _metaSpan(text) {
    var s = document.createElement('span');
    s.className = 'wishlist-meta-item';
    s.textContent = text;
    return s;
  }

  function _actionLabel(text) {
    var s = document.createElement('span');
    s.className = 'wishlist-cta-label text-muted';
    s.textContent = text;
    return s;
  }

  // CTA: open the existing HF browser pre-filtered by the model id so the
  // user can confirm + download. Falls back to a manual "Add to swarm" hint
  // when the HF browser isn't available.
  function _onHelpHost(entry) {
    if (App.modelBrowser && typeof App.modelBrowser.openWithQuery === 'function') {
      App.modelBrowser.openWithQuery(entry.display_name || entry.model_id);
    } else {
      App.ui.showBanner('info', I18n.t('wishlist.cta_open_hf_hint'));
    }
  }

  function _renderWishlist(snapshot) {
    var listEl = document.getElementById('wishlist-list');
    var metaEl = document.getElementById('wishlist-meta');
    if (!listEl) return;
    listEl.innerHTML = '';
    var entries = (snapshot && snapshot.entries) || [];
    if (entries.length === 0) {
      var empty = document.createElement('div');
      empty.className = 'text-muted';
      empty.textContent = I18n.t('wishlist.empty');
      listEl.appendChild(empty);
    } else {
      entries.forEach(function (e) { listEl.appendChild(_renderEntry(e)); });
    }
    if (metaEl && snapshot && snapshot.computed_at) {
      var ageSec = Math.max(0, Math.floor(Date.now() / 1000 - snapshot.computed_at));
      metaEl.textContent = I18n.t('wishlist.updated', { secs: ageSec });
    }
  }

  function _renderRunning(capacity) {
    var grid = document.getElementById('swarm-running-grid');
    if (!grid) return;
    grid.innerHTML = '';
    var serveable = (capacity && capacity.serveable_models) || [];
    if (serveable.length === 0) {
      var empty = document.createElement('div');
      empty.className = 'text-muted';
      empty.textContent = I18n.t('swarm.running_empty');
      grid.appendChild(empty);
      return;
    }
    serveable.forEach(function (m) {
      var card = document.createElement('div');
      card.className = 'capacity-card' + (m.hosted_by_us ? ' capacity-card-mine' : '');
      var name = document.createElement('div');
      name.className = 'capacity-card-name';
      name.textContent = m.display_name || m.model_id;
      card.appendChild(name);
      var meta = document.createElement('div');
      meta.className = 'capacity-card-meta text-muted text-2xs';
      meta.textContent = I18n.t('wishlist.meta_size', { size: _humaniseSize(m.size_mb) }) +
        ' · ' + I18n.t('wishlist.meta_replicas_ok', { have: m.holders });
      card.appendChild(meta);
      if (m.hosted_by_us) {
        var badge = document.createElement('span');
        badge.className = 'capacity-card-badge';
        badge.textContent = I18n.t('capacity.you_host_this');
        card.appendChild(badge);
      }
      grid.appendChild(card);
    });
  }

  function _switchSubtab(name) {
    document.querySelectorAll('.swarm-subtab').forEach(function (b) {
      b.classList.toggle('active', b.dataset.swarmSubtab === name);
    });
    ['wishlist', 'running', 'search', 'capacity'].forEach(function (n) {
      var pane = document.getElementById('swarm-subview-' + n);
      if (pane) pane.style.display = n === name ? '' : 'none';
    });
  }

  App.swarmTab = {
    /** Called from notifications.js whenever a stats_update WS frame arrives. */
    onStats: function (data) {
      if (data && data.wishlist) _renderWishlist(data.wishlist);
      if (data && data.swarm_capacity) _renderRunning(data.swarm_capacity);
    },

    /** Called from init.js after the tab buttons render. */
    bind: function () {
      // Subtab switching
      document.querySelectorAll('.swarm-subtab').forEach(function (b) {
        b.addEventListener('click', function () {
          _switchSubtab(b.dataset.swarmSubtab);
        });
      });
      // Search-subtab "Open HF browser" button
      var openHfBtn = document.getElementById('swarm-open-hf-browser');
      if (openHfBtn) {
        openHfBtn.addEventListener('click', function () {
          if (App.modelBrowser && typeof App.modelBrowser.open === 'function') {
            App.modelBrowser.open();
          } else if (typeof App.ui !== 'undefined') {
            App.ui.showBanner('info', I18n.t('wishlist.cta_open_hf_hint'));
          }
        });
      }
      // Initial fetch in case the user lands on this tab before the first
      // stats_update lands.
      _refreshFromRest();
    },

    /** Called when the user switches to the Swarm tab. */
    onShow: function () {
      _refreshFromRest();
    },
  };

  function _refreshFromRest() {
    if (!App.authFetch) return;
    App.authFetch('/api/admin/wishlist').then(function (r) {
      if (!r.ok) return null;
      return r.json();
    }).then(function (data) {
      if (data) _renderWishlist(data);
    }).catch(function () { /* non-fatal */ });
    App.authFetch('/api/admin/swarm/capacity').then(function (r) {
      if (!r.ok) return null;
      return r.json();
    }).then(function (data) {
      if (data) _renderRunning(data);
    }).catch(function () { /* non-fatal */ });
    App.authFetch('/api/admin/swarm/capacity-plan').then(function (r) {
      if (!r.ok) return null;
      return r.json();
    }).then(function (data) {
      if (data) _renderCapacityPlan(data);
    }).catch(function () { /* non-fatal */ });
  }

  // R113: render the Capacity Plan / What-If view. Three baked scenarios
  // (small / medium / large) + a headline_target. Educational copy:
  // "if N more contributors joined with X GB each, you'd unlock Y" —
  // makes the contribution → capability link tangible for non-technical
  // users.
  function _renderCapacityPlan(plan) {
    var container = document.getElementById('capacity-plan-content');
    if (!container) return;
    container.innerHTML = '';

    if (plan.headline_target) {
      var t = plan.headline_target;
      var hero = document.createElement('div');
      hero.className = 'capacity-plan-hero';
      hero.innerHTML =
        '<div class="capacity-plan-hero-icon">&#127919;</div>' +
        '<div class="capacity-plan-hero-body">' +
        '<div class="capacity-plan-hero-title">' +
        I18n.t('capacity_plan.hero_title', { name: U.escapeHtml(t.display_name) }) +
        '</div>' +
        '<div class="capacity-plan-hero-msg">' +
        I18n.t('capacity_plan.hero_msg', {
          contributors: t.contributors_needed,
          shortfall: _humaniseSize(t.vram_shortfall_mb),
        }) +
        '</div>' +
        '</div>';
      container.appendChild(hero);
    }

    var grid = document.createElement('div');
    grid.className = 'capacity-plan-grid';
    (plan.scenarios || []).forEach(function (sc) {
      var card = document.createElement('div');
      card.className = 'capacity-plan-card' + (sc.unlocks_anything ? ' capacity-plan-card-active' : '');

      var title = document.createElement('div');
      title.className = 'capacity-plan-card-title';
      title.textContent = I18n.t('capacity_plan.scenario_' + sc.label);
      card.appendChild(title);

      var sub = document.createElement('div');
      sub.className = 'capacity-plan-card-sub text-muted text-2xs';
      sub.textContent = I18n.t('capacity_plan.scenario_sub', {
        nodes: sc.added_nodes,
        gb: sc.vram_gb_per_node,
      });
      card.appendChild(sub);

      var projected = document.createElement('div');
      projected.className = 'capacity-plan-projected';
      projected.innerHTML =
        '<span class="text-muted">' + I18n.t('capacity_plan.projected_total') +
        '</span> <strong>' + _humaniseSize(sc.projected_total_vram_mb) + '</strong>';
      card.appendChild(projected);

      if (sc.unlocks_anything && sc.newly_unlocked.length > 0) {
        var unlockTitle = document.createElement('div');
        unlockTitle.className = 'capacity-plan-unlocks-title text-2xs mt-1';
        unlockTitle.textContent = I18n.t('capacity_plan.unlocks');
        card.appendChild(unlockTitle);
        var ul = document.createElement('ul');
        ul.className = 'capacity-plan-unlocks-list';
        sc.newly_unlocked.forEach(function (m) {
          var li = document.createElement('li');
          li.textContent = m.display_name + ' (' + _humaniseSize(m.size_mb) + ')';
          ul.appendChild(li);
        });
        card.appendChild(ul);
      } else {
        var none = document.createElement('div');
        none.className = 'capacity-plan-no-unlock text-muted text-2xs mt-1';
        none.textContent = I18n.t('capacity_plan.no_new_unlock');
        card.appendChild(none);
      }
      grid.appendChild(card);
    });
    container.appendChild(grid);
  }
})();
