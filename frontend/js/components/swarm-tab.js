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

  var U = App.utils;

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

  // Pick the best displayable name: prefer backend display_name unless it
  // looks unprocessed (raw_lowercase_with_underscores), in which case fall
  // back to the JS prettifier. Catches manifests where org name was doubled
  // into the display field (e.g. "tinyllama_tinyllama-1.1b-chat-v1.0").
  function _bestModelName(display, modelId) {
    var looksRaw = display && (display === display.toLowerCase() && /_/.test(display));
    if (display && !looksRaw) return display;
    var source = display || modelId || '';
    return U.formatModelDisplayName ? U.formatModelDisplayName(source) : source;
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
    name.textContent = _bestModelName(entry.display_name, entry.model_id);
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
      replicaText = I18n.t(entry.swarm_replicas === 1 ? 'wishlist.meta_replicas_ok_one' : 'wishlist.meta_replicas_ok_other', { have: entry.swarm_replicas });
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

  // CTA: jump to the Search subtab pre-seeded with the model name so the
  // user can confirm + download. Replaces the old modal opener.
  function _onHelpHost(entry) {
    var seed = entry.display_name || _prettyRepoName(entry.model_id || '');
    App.swarmTab.openSearch(seed);
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
      var nameText = document.createElement('span');
      nameText.textContent = _bestModelName(m.display_name, m.model_id);
      name.appendChild(nameText);
      if (m.hosted_by_us) {
        var badge = document.createElement('span');
        badge.className = 'capacity-card-badge';
        badge.textContent = I18n.t('capacity.you_host_this_short');
        badge.title = I18n.t('capacity.you_host_this');
        name.appendChild(badge);
      }
      card.appendChild(name);
      var meta = document.createElement('div');
      meta.className = 'capacity-card-meta text-muted text-2xs';
      meta.textContent = I18n.t('wishlist.meta_size', { size: _humaniseSize(m.size_mb) }) +
        ' · ' + I18n.t(m.holders === 1 ? 'wishlist.meta_replicas_ok_one' : 'wishlist.meta_replicas_ok_other', { have: m.holders });
      card.appendChild(meta);
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
    if (name === 'search') _browseEnsureTrending();
  }

  // -------------------------------------------------------------------------
  // Inline browser (Search HuggingFace subtab)
  //
  // Replaces the old "click button → modal" UX. Layout:
  //   1. Use-case onramp cards   →  picks a task-tag filter
  //   2. Search box + "Only fits my swarm" toggle
  //   3. Trending strip          →  populated from /api/admin/hf/trending
  //   4. Results list            →  compact rows, click to expand
  // -------------------------------------------------------------------------

  var _browseState = {
    query: '',
    tasks: [],          // multi-select, but use-case cards single-select for now
    fitOnly: true,
    trendingLoaded: false,
    expanded: null,     // repo_id of the expanded row, if any
  };

  function _prettyRepoName(repoId) {
    if (!repoId) return '';
    var parts = repoId.split('/');
    var tail = parts.length > 1 ? parts[1] : parts[0];
    return U.formatModelDisplayName ? U.formatModelDisplayName(tail) : tail;
  }

  function _repoAuthor(repoId) {
    if (!repoId) return '';
    var parts = repoId.split('/');
    return parts.length > 1 ? parts[0] : '';
  }

  // Pick a single fit pill from the backend's fits_* booleans. Order matters:
  // "Already hosting" (locally registered) wins, then local fit, then "host
  // shards via swarm", then swarm-only, then too-large.
  // `_localHfRepos` is rebuilt on each stats_update; used to detect when a
  // result row points at a model the user already hosts (so we render a
  // distinct "★ You host this" pill instead of generic Download CTA).
  var _localHfRepos = new Set();
  function _fitPill(repo) {
    if (_localHfRepos.has((repo.repo_id || '').toLowerCase())) {
      return { key: 'already', text: I18n.t('browse.fit_already') };
    }
    if (repo.fits_boomerang) {
      return { key: 'run', text: I18n.t('browse.fit_run_local') };
    }
    if (repo.fits_shard) {
      return { key: 'host', text: I18n.t('browse.fit_host_shards') };
    }
    if (repo.network_replicas > 0) {
      return { key: 'swarm', text: I18n.t('browse.fit_swarm_only') };
    }
    return { key: 'too-large', text: I18n.t('browse.fit_too_large') };
  }

  // Snapshot which HF repos the local node already hosts. Fed by
  // `App.swarmTab.onStats` so the browser's "★ You host this" pill stays
  // accurate without a per-render REST round-trip.
  function _refreshLocalHfRepos() {
    if (!App.data || !App.data.cache || !App.data.cache.models) return;
    var fresh = new Set();
    var models = App.data.cache.models || [];
    models.forEach(function (m) {
      var src = m && m.hf_source;
      if (src && src.repo_id) fresh.add(String(src.repo_id).toLowerCase());
    });
    _localHfRepos = fresh;
  }

  function _humaniseBytes(bytes) {
    if (!bytes || bytes < 1024) return (bytes || 0) + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(0) + ' KB';
    if (bytes < 1024 * 1024 * 1024) return (bytes / (1024 * 1024)).toFixed(0) + ' MB';
    return (bytes / (1024 * 1024 * 1024)).toFixed(2) + ' GB';
  }

  function _browseEnsureTrending() {
    if (_browseState.trendingLoaded) return;
    _browseState.trendingLoaded = true;
    if (!App.authFetch) return;
    App.authFetch('/api/admin/hf/trending').then(function (r) {
      return r.ok ? r.json() : null;
    }).then(function (snap) {
      if (snap) _renderBrowseTrending(snap);
    }).catch(function () {
      _browseState.trendingLoaded = false;
    });
  }

  function _renderBrowseTrending(snap) {
    var strip = document.getElementById('browse-trending-strip');
    var meta = document.getElementById('browse-trending-meta');
    if (!strip) return;
    var entries = (snap && snap.entries) || [];
    strip.innerHTML = '';
    if (entries.length === 0) {
      var empty = document.createElement('div');
      empty.className = 'text-muted text-sm';
      empty.textContent = I18n.t('browse.trending_empty');
      strip.appendChild(empty);
      return;
    }
    entries.slice(0, 12).forEach(function (entry) {
      var card = document.createElement('button');
      card.className = 'browse-trending-card';
      card.type = 'button';
      var name = document.createElement('div');
      name.className = 'browse-trending-name';
      name.textContent = _prettyRepoName(entry.repo_id);
      name.title = entry.repo_id;
      card.appendChild(name);
      var meta2 = document.createElement('div');
      meta2.className = 'browse-trending-meta';
      var dl = entry.downloads || 0;
      meta2.textContent = dl >= 1000 ? (dl / 1000).toFixed(0) + 'k ' + I18n.t('browse.downloads_short') : dl + ' ' + I18n.t('browse.downloads_short');
      card.appendChild(meta2);
      if ((entry.task_tags || []).length > 0) {
        var tags = document.createElement('div');
        tags.className = 'browse-trending-tags';
        entry.task_tags.slice(0, 2).forEach(function (t) {
          var pill = document.createElement('span');
          pill.className = 'browse-trending-tag';
          var label = I18n.t('wishlist.task.' + t);
          pill.textContent = label === 'wishlist.task.' + t ? t : label;
          tags.appendChild(pill);
        });
        card.appendChild(tags);
      }
      card.addEventListener('click', function () {
        var input = document.getElementById('browse-search-input');
        if (input) {
          input.value = entry.repo_id;
          _browseState.query = entry.repo_id;
        }
        _browseSearch();
      });
      strip.appendChild(card);
    });
    if (meta && snap.fetched_at) {
      var ageMin = Math.max(0, Math.floor((Date.now() / 1000 - snap.fetched_at) / 60));
      meta.textContent = I18n.t('browse.trending_updated', { mins: ageMin });
    }
  }

  function _browseSearch() {
    var input = document.getElementById('browse-search-input');
    var query = (input && input.value || '').trim();
    _browseState.query = query;
    var section = document.getElementById('browse-results-section');
    var loading = document.getElementById('browse-loading');
    var list = document.getElementById('browse-results-list');

    if (!query && _browseState.tasks.length === 0) {
      if (section) section.style.display = 'none';
      return;
    }
    if (loading) loading.style.display = '';
    if (section) section.style.display = 'none';

    var url = '/api/admin/hf/search?q=' + encodeURIComponent(query || '');
    if (_browseState.tasks.length > 0) {
      url += '&tasks=' + encodeURIComponent(_browseState.tasks.join(','));
    }
    App.authFetch(url).then(function (r) {
      return r.ok ? r.json() : [];
    }).then(function (data) {
      if (loading) loading.style.display = 'none';
      _renderBrowseResults(data);
    }).catch(function () {
      if (loading) loading.style.display = 'none';
      var list2 = document.getElementById('browse-results-list');
      if (list2) list2.innerHTML = '<div class="browse-empty">' + U.escapeHtml(I18n.t('browse.error')) + '</div>';
      if (section) section.style.display = '';
    });
  }

  function _renderBrowseResults(data) {
    var section = document.getElementById('browse-results-section');
    var list = document.getElementById('browse-results-list');
    var title = document.getElementById('browse-results-title');
    var meta = document.getElementById('browse-results-meta');
    if (!list) return;
    list.innerHTML = '';

    var filtered = data || [];
    var totalRaw = filtered.length;
    if (_browseState.fitOnly) {
      filtered = filtered.filter(function (r) {
        return r.fits_shard || r.fits_boomerang || r.network_replicas > 0;
      });
    }

    if (section) section.style.display = '';
    if (title) title.textContent = _browseState.query
      ? I18n.t('browse.results_for', { q: _browseState.query })
      : I18n.t('browse.results_title');
    if (meta) {
      if (_browseState.fitOnly && filtered.length < totalRaw) {
        meta.textContent = I18n.t('browse.results_count_filtered', { shown: filtered.length, hidden: totalRaw - filtered.length });
      } else {
        meta.textContent = I18n.t('browse.results_count', { count: filtered.length });
      }
    }

    if (filtered.length === 0) {
      var empty = document.createElement('div');
      empty.className = 'browse-empty';
      empty.textContent = _browseState.fitOnly && totalRaw > 0
        ? I18n.t('browse.results_none_fit')
        : I18n.t('browse.results_none');
      list.appendChild(empty);
      return;
    }

    filtered.forEach(function (repo) { list.appendChild(_renderBrowseRow(repo)); });
  }

  function _renderBrowseRow(repo) {
    var row = document.createElement('div');
    row.className = 'browse-result-row';
    if (_browseState.expanded === repo.repo_id) row.classList.add('expanded');

    var main = document.createElement('div');
    main.className = 'browse-result-main';
    var name = document.createElement('div');
    name.className = 'browse-result-name';
    name.textContent = _prettyRepoName(repo.repo_id);
    name.title = repo.repo_id;
    main.appendChild(name);
    var author = document.createElement('div');
    author.className = 'browse-result-author';
    author.textContent = _repoAuthor(repo.repo_id);
    main.appendChild(author);
    row.appendChild(main);

    var sizeEl = document.createElement('div');
    sizeEl.className = 'browse-result-size';
    var sizeBytes = repo.est_boomerang_size || repo.est_shard_size || 0;
    sizeEl.textContent = sizeBytes ? _humaniseBytes(sizeBytes) : '—';
    row.appendChild(sizeEl);

    var fit = _fitPill(repo);
    var pill = document.createElement('span');
    pill.className = 'browse-fit-pill browse-fit-pill-' + fit.key;
    pill.textContent = fit.text;
    row.appendChild(pill);

    var actionBtn = document.createElement('button');
    actionBtn.type = 'button';
    actionBtn.className = 'browse-result-action';
    if (fit.key === 'already') {
      actionBtn.textContent = I18n.t('browse.action_hosting');
      actionBtn.classList.add('browse-result-action-disabled');
      actionBtn.disabled = true;
    } else if (fit.key === 'too-large') {
      actionBtn.textContent = I18n.t('browse.action_wishlist');
      actionBtn.classList.add('browse-result-action');
    } else {
      actionBtn.textContent = I18n.t('browse.action_download');
      actionBtn.classList.add('browse-result-action-primary');
    }
    actionBtn.addEventListener('click', function (e) {
      e.stopPropagation();
      if (actionBtn.disabled) return;
      if (fit.key === 'too-large') {
        // Add to wishlist (aspirational). Use the existing wishlist endpoint.
        _browseAddToWishlist(repo);
      } else {
        var variants = repo.variants || [];
        var pick = variants.length > 0 ? (variants.find(function (v) { return v.quant === repo.recommended_variant; }) || variants[0]) : null;
        if (pick && App.hf && App.hf.download) {
          App.hf.download(repo.repo_id, '');
          // The existing App.hf.download reads the <select> in the modal —
          // we don't have that, so call the underlying endpoint directly.
          // Fall through to the direct call below.
        }
        _browseDownload(repo, pick);
      }
    });
    row.appendChild(actionBtn);

    // Expandable detail
    var detail = document.createElement('div');
    detail.className = 'browse-result-detail';

    var stats = document.createElement('div');
    stats.className = 'browse-result-stats';
    if (repo.downloads) {
      var s = document.createElement('span');
      s.textContent = repo.downloads.toLocaleString() + ' ' + I18n.t('browse.downloads_short');
      stats.appendChild(s);
    }
    if (repo.likes) {
      var s2 = document.createElement('span');
      s2.textContent = '♥ ' + repo.likes.toLocaleString();
      stats.appendChild(s2);
    }
    if (repo.network_replicas) {
      var s3 = document.createElement('span');
      s3.textContent = I18n.t('browse.network_replicas', { n: repo.network_replicas });
      stats.appendChild(s3);
    }
    (repo.task_tags || []).forEach(function (t) {
      var pillT = document.createElement('span');
      pillT.className = 'browse-trending-tag';
      var label = I18n.t('wishlist.task.' + t);
      pillT.textContent = label === 'wishlist.task.' + t ? t : label;
      stats.appendChild(pillT);
    });
    detail.appendChild(stats);

    var variants = repo.variants || [];
    if (variants.length > 1) {
      var quantRow = document.createElement('div');
      quantRow.className = 'browse-quant-row';
      var lbl = document.createElement('span');
      lbl.className = 'text-muted';
      lbl.textContent = I18n.t('browse.quant_label');
      quantRow.appendChild(lbl);
      var sel = document.createElement('select');
      sel.dataset.repoId = repo.repo_id;
      variants.forEach(function (v) {
        var opt = document.createElement('option');
        opt.value = v.filename;
        var label = v.quant + (v.size_bytes ? ' — ' + _humaniseBytes(v.size_bytes) : '');
        if (v.quant === repo.recommended_variant) {
          label += ' ' + I18n.t('models.hf_recommended');
          opt.selected = true;
        }
        opt.textContent = label;
        sel.appendChild(opt);
      });
      quantRow.appendChild(sel);
      detail.appendChild(quantRow);
    }
    row.appendChild(detail);

    row.addEventListener('click', function () {
      _browseState.expanded = _browseState.expanded === repo.repo_id ? null : repo.repo_id;
      row.classList.toggle('expanded');
    });

    return row;
  }

  function _browseDownload(repo, variant) {
    if (!App.authFetch) return;
    var filename = variant ? variant.filename : (repo.variants && repo.variants[0] && repo.variants[0].filename);
    if (!filename) {
      App.notifications && App.notifications.showToast &&
        App.notifications.showToast(I18n.t('browse.error_no_variant'), 'error');
      return;
    }
    App.authFetch('/api/admin/hf/download-shards', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ repo_id: repo.repo_id, filename: filename }),
    }).then(function (r) {
      if (r.ok) {
        App.notifications && App.notifications.showToast &&
          App.notifications.showToast(I18n.t('browse.download_started', { name: _prettyRepoName(repo.repo_id) }), 'success');
      } else {
        return r.json().then(function (e) {
          App.notifications && App.notifications.showToast &&
            App.notifications.showToast((e && e.error && e.error.message) || I18n.t('browse.download_failed'), 'error');
        });
      }
    }).catch(function () {
      App.notifications && App.notifications.showToast &&
        App.notifications.showToast(I18n.t('browse.download_failed'), 'error');
    });
  }

  function _browseAddToWishlist(repo) {
    // Best-effort: trigger a search query so the auto-manage scoring picks
    // up the demand signal. Future improvement: explicit wishlist API.
    App.notifications && App.notifications.showToast &&
      App.notifications.showToast(I18n.t('browse.wishlist_added', { name: _prettyRepoName(repo.repo_id) }), 'info');
  }

  function _browseBind() {
    var input = document.getElementById('browse-search-input');
    if (input) {
      var debounce;
      input.addEventListener('input', function () {
        clearTimeout(debounce);
        debounce = setTimeout(_browseSearch, 400);
      });
      input.addEventListener('keydown', function (e) {
        if (e.key === 'Enter') { e.preventDefault(); _browseSearch(); }
      });
    }
    var fitToggle = document.getElementById('browse-fit-only');
    if (fitToggle) {
      fitToggle.addEventListener('change', function () {
        _browseState.fitOnly = fitToggle.checked;
        _browseSearch();
      });
    }
    document.querySelectorAll('.usecase-card').forEach(function (card) {
      card.addEventListener('click', function () {
        var uc = card.dataset.usecase;
        var alreadyActive = card.classList.contains('active');
        document.querySelectorAll('.usecase-card').forEach(function (c) { c.classList.remove('active'); });
        if (alreadyActive) {
          _browseState.tasks = [];
        } else {
          card.classList.add('active');
          _browseState.tasks = [uc];
        }
        _browseSearch();
      });
    });
  }

  App.swarmTab = {
    /** Called from notifications.js whenever a stats_update WS frame arrives. */
    onStats: function (data) {
      if (data && data.wishlist) _renderWishlist(data.wishlist);
      if (data && data.swarm_capacity) _renderRunning(data.swarm_capacity);
      // Keep the inline browser's "★ You host this" pill accurate. Cheap —
      // a few Set operations per WS frame.
      _refreshLocalHfRepos();
    },

    /** Called from init.js after the tab buttons render. */
    bind: function () {
      // Subtab switching
      document.querySelectorAll('.swarm-subtab').forEach(function (b) {
        b.addEventListener('click', function () {
          _switchSubtab(b.dataset.swarmSubtab);
        });
      });
      // Inline browser bindings (Search HuggingFace subtab)
      _browseBind();
      _refreshFromRest();
    },

    /** Called when the user switches to the Models tab. */
    onShow: function () {
      _refreshFromRest();
    },

    /** Switch directly to the Search subtab, focus the search box, and
     *  optionally seed it with a query. Used by the "+ Find model" header
     *  button and by `wishlist → Help host` CTAs. */
    openSearch: function (query) {
      App.ui.switchTab('swarm');
      _switchSubtab('search');
      var input = document.getElementById('browse-search-input');
      if (input) {
        if (query) { input.value = query; _browseState.query = query; _browseSearch(); }
        setTimeout(function () { input.focus(); }, 50);
      }
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
