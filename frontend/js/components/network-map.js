'use strict';

// ============================================================================
// SwarmLLM — Network Map Component
// SVG world heatmap with TopoJSON, tooltips, regional health
// ============================================================================

(function() {
  var U = App.utils;

  App.networkMap = {
    data: null,
    mapRendered: false,

    // Apply heatmap fill/stroke styling to all known region paths.
    // `countsByCode` is { 'US': n, 'JP': n, ... }; missing codes get the zero style.
    _applyRegionColors: function(countsByCode, maxCount) {
      var allCodes = Object.keys(App.networkMap.paths);
      for (var j = 0; j < allCodes.length; j++) {
        var c = allCodes[j];
        var el = document.getElementById('region-' + c);
        if (!el) continue;
        var n = countsByCode[c] || 0;
        if (n === 0) {
          el.style.fill = 'rgba(59,130,246,0.04)';
          el.style.stroke = 'rgba(59,130,246,0.3)';
          el.style.strokeWidth = '0.5';
          el.removeAttribute('filter');
        } else {
          // Lit by its FILL, with a border as thin as its neighbours'. A
          // border is drawn centred on the line two countries share, so a
          // thick one — and the 3-unit glow it had — lay over the neighbour:
          // a lit India read as spilling into Myanmar and Bangladesh.
          var intensity = Math.max(0.25, n / Math.max(maxCount, 1));
          var fillAlpha = (0.14 + intensity * 0.3).toFixed(2);
          var strokeAlpha = (0.55 + intensity * 0.45).toFixed(2);
          el.style.fill = 'rgba(59,130,246,' + fillAlpha + ')';
          el.style.stroke = 'rgba(100,180,255,' + strokeAlpha + ')';
          el.style.strokeWidth = '0.7';
          el.removeAttribute('filter');
        }
      }
    },

    // THE MAP — `/static/data/countries.json`, built from Natural Earth
    // (public domain) by `examples/build_map_data.sh`: TopoJSON already
    // PROJECTED (Equal Earth, areas true) and fitted to a 1000-pixel frame
    // with its origin at the top left, so it is drawn as it is and this file
    // does no map arithmetic. Prior art: us-atlas ships pre-projected files for
    // the same reason, and mapshaper's `-o width=` writes them. This file once
    // projected by hand and started a new piece of a ring wherever it crossed
    // the 180th meridian — wrong for a FILLED shape, which mapshaper's `-proj`
    // cuts and closes properly. Objects:
    //
    // - `default`, `IN` — the outlines, one object per BORDER VIEW
    //   (`worldviewFor`): Natural Earth's de facto borders, and its India
    //   edition — India's official boundaries (all of Jammu and Kashmir and
    //   Aksai Chin as India), which a viewer in India expects and the de facto
    //   map cuts short. Prior art: Google Maps shows each country its own
    //   borders; Mapbox calls it a `worldview`, HERE a geopolitical view. Each
    //   outline carries `c`, its two-letter code (null for an area no code
    //   names). The map read world-atlas's outlines through a numeric-id table
    //   typed by hand here, which lacked nine — Slovenia, Serbia, Armenia,
    //   Azerbaijan, North Macedonia, Montenegro, Kosovo, Eswatini, Puerto Rico
    //   — so their computers never lit the map.
    // - `points` — each code's label point (Natural Earth's LABEL_X / LABEL_Y,
    //   placed by hand inside a country's main body). A computer's dot and
    //   every route arc start there. They started at the middle of the
    //   OUTLINE's bounding box, which for a country with land far apart is open
    //   sea or another country: Russia's and Fiji's reach both sides of the
    //   180th meridian, so their middle was 0° (Russia's dot in Norway, Fiji's
    //   in Africa); France's box takes in French Guiana (a dot in the
    //   Atlantic). It is also how a country too small to keep an outline
    //   (Vatican City, Singapore, Malta) gets a dot.
    // - `frame`, `grid`, `equator` — the map's edge (cut at 84° N and 58° S:
    //   Antarctica is not drawn, nor the empty band below Cape Horn) and a 30°
    //   graticule.
    //
    // Guard: `every_country_the_map_draws_has_a_code_and_a_marker`.
    points: {},
    worldview: null,
    _shapes: null,
    _loadShapes: async function() {
      if (App.networkMap._shapes) return App.networkMap._shapes;
      var topo = await fetch('/static/data/countries.json').then(function(r) { return r.json(); });
      var at = {};
      topojson.feature(topo, topo.objects.points).features.forEach(function(f) {
        if (f.geometry && f.properties && f.properties.c) at[f.properties.c] = f.geometry.coordinates;
      });
      // The frame's margin is the same on every side, so the canvas is the
      // drawing's far corner plus its near one.
      var b = topojson.bbox(topo);
      App.networkMap._shapes = { topo: topo, at: at, w: Math.ceil(b[2] + b[0]), h: Math.ceil(b[3] + b[1]) };
      App.networkMap.points = at;
      return App.networkMap._shapes;
    },

    // WHOSE BORDERS TO DRAW: the view named for this computer's country
    // (`self_region`), else for the browser's region (`en-IN`, `hi-IN`), when
    // the outlines carry one; otherwise Natural Earth's de facto borders.
    worldviewFor: function(selfRegion) {
      var objs = (App.networkMap._shapes && App.networkMap._shapes.topo.objects) || {};
      var asked = [selfRegion, App.networkMap._browserRegion()];
      for (var i = 0; i < asked.length; i++) {
        if (asked[i] && asked[i] !== 'default' && objs[asked[i]]) return asked[i];
      }
      return 'default';
    },
    _browserRegion: function() {
      try {
        var langs = navigator.languages && navigator.languages.length ? navigator.languages : [navigator.language];
        for (var i = 0; i < langs.length; i++) {
          var m = /-([A-Za-z]{2})(?:-|$)/.exec(langs[i] || '');
          if (m) return m[1].toUpperCase();
        }
      } catch (e) {}
      return null;
    },

    // The frame, a 30° graticule and the equator, as the file draws them. The
    // old straight grid every 50 px matched no round number of degrees.
    _graticuleSvg: function(topo) {
      var d = function(name) { return App.networkMap._objectPath(topo, name); };
      return '<path d="' + d('frame') + '" fill="var(--accent)" fill-opacity="0.025" stroke="var(--accent)" stroke-width="0.5" stroke-opacity="0.3"/>'
        + '<path d="' + d('grid') + '" fill="none" stroke="var(--accent)" stroke-width="0.3" stroke-opacity="0.15"/>'
        + '<path d="' + d('equator') + '" fill="none" stroke="var(--accent)" stroke-width="0.5" stroke-opacity="0.18" stroke-dasharray="8,4"/>';
    },

    // One object of the file as SVG path data.
    _objectPath: function(topo, name) {
      if (!topo.objects[name]) return '';
      return topojson.feature(topo, topo.objects[name]).features
        .map(function(f) { return App.networkMap.geomToPath(f.geometry); }).join('');
    },

    // A line through points already in the map's pixels; `closed` for a ring.
    lineToPath: function(pts, closed) {
      var d = '';
      for (var i = 0; i < pts.length; i++) {
        d += (i ? 'L' : 'M') + (Math.round(pts[i][0] * 10) / 10) + ',' + (Math.round(pts[i][1] * 10) / 10);
      }
      return pts.length && closed ? d + 'Z' : d;
    },

    // A geometry as SVG path data. `null` — topojson-client's answer for an
    // outline with no shape — is drawn as nothing; read as a shape, one threw
    // and every outline after it went undrawn.
    geomToPath: function(geom) {
      var nm = App.networkMap;
      if (!geom) return '';
      var rings = function(list) { return list.map(function(r) { return nm.lineToPath(r, true); }).join(''); };
      switch (geom.type) {
        case 'Polygon': return rings(geom.coordinates);
        case 'MultiPolygon': return geom.coordinates.map(rings).join('');
        case 'LineString': return nm.lineToPath(geom.coordinates, false);
        case 'MultiLineString': return geom.coordinates.map(function(l) { return nm.lineToPath(l, false); }).join('');
        default: return '';
      }
    },

    paths: {},

    buildSvg: async function(view) {
      var container = document.getElementById('world-map');
      if (!container) return;

      // Without the file the map is an empty canvas of the same shape, and
      // the stats line under it still counts.
      App.networkMap.paths = {};
      var unnamed = '';
      var lines = '';
      var w = 1000, h = 440;
      view = view || 'default';
      try {
        var shapes = await App.networkMap._loadShapes();
        w = shapes.w;
        h = shapes.h;
        lines = App.networkMap._graticuleSvg(shapes.topo);
        var objs = shapes.topo.objects;
        if (!objs[view]) view = 'default';
        var features = topojson.feature(shapes.topo, objs[view]).features;
        for (var i = 0; i < features.length; i++) {
          var code = features[i].properties && features[i].properties.c;
          var d = App.networkMap.geomToPath(features[i].geometry);
          // An area no code names (a disputed glacier, a buffer zone) is
          // drawn and never lit. Two outlines can share a code (Northern
          // Cyprus with Cyprus, Somaliland with Somalia — what an address
          // lookup answers for a computer there): one element each, or the
          // ids would collide.
          if (!code) unnamed += d;
          else App.networkMap.paths[code] = (App.networkMap.paths[code] || '') + d;
        }
      } catch (e) {
      }
      App.networkMap.worldview = view;

      var svg = '<svg viewBox="0 0 ' + w + ' ' + h + '" xmlns="http://www.w3.org/2000/svg" class="world-svg">';
      svg += '<defs>';
      svg += '<filter id="glow-sm"><feGaussianBlur stdDeviation="1.5" result="blur"/><feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge></filter>';
      svg += '<filter id="glow-md"><feGaussianBlur stdDeviation="3" result="blur"/><feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge></filter>';
      svg += '</defs>';
      svg += lines;
      if (unnamed) svg += '<path d="' + unnamed + '" fill="rgba(59,130,246,0.04)" stroke="rgba(59,130,246,0.3)" stroke-width="0.5"/>';
      var codes = Object.keys(App.networkMap.paths);
      for (var i = 0; i < codes.length; i++) {
        var code = codes[i];
        var d = App.networkMap.paths[code];
        if (!d) continue;
        svg += '<path id="region-' + code + '" d="' + d + '" fill="rgba(59,130,246,0.04)" stroke="rgba(59,130,246,0.3)" stroke-width="0.5" class="map-region" data-code="' + code + '"/>';
      }
      svg += '</svg>';
      container.innerHTML = svg;

      container.querySelectorAll('.map-region').forEach(function(el) {
        App.networkMap._hoverShows(el, el.dataset.code);
      });

      App.networkMap.mapRendered = true;
    },

    _hoverShows: function(el, code) {
      el.addEventListener('mouseenter', function(e) { App.networkMap.showTooltip(e, code); });
      el.addEventListener('mousemove', function(e) { App.networkMap.moveTooltip(e); });
      el.addEventListener('mouseleave', function() { App.networkMap.hideTooltip(); });
    },

    // A ring round THIS computer's country, so a person can check the map
    // against where they are. Named in the key (`#map-legend`), which lists
    // every mark the map paints. `selfRegion` is the code the daemon placed
    // this computer in (`api::admin::map_region_of_this_computer`), `??`
    // when its country is not known — then there is no ring, as there is no
    // dot.
    renderThisComputer: function(svg, selfRegion) {
      svg.querySelectorAll('.map-this-computer').forEach(function(el) { el.remove(); });
      var key = document.getElementById('map-legend-self');
      var at = App.networkMap.regionCenter(selfRegion);
      if (key) key.style.display = at ? '' : 'none';
      if (!at) return;
      var ring = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
      ring.setAttribute('cx', at[0]);
      ring.setAttribute('cy', at[1]);
      ring.setAttribute('r', 11);
      ring.setAttribute('class', 'map-this-computer');
      var t = document.createElementNS('http://www.w3.org/2000/svg', 'title');
      t.textContent = I18n.t('chat.status.on_this_computer');
      ring.appendChild(t);
      svg.appendChild(ring);
    },

    // The data first: whose borders to draw depends on where this computer
    // is (`worldviewFor`), and the outlines are drawn again if that changed.
    refresh: async function() {
      var data = null;
      try {
        data = await (await App.authFetch('/api/admin/network-map')).json();
      } catch (e) {}
      try { await App.networkMap._loadShapes(); } catch (e) {}
      var view = App.networkMap.worldviewFor(data && data.self_region);
      if (!App.networkMap.mapRendered || App.networkMap.worldview !== view) {
        await App.networkMap.buildSvg(view);
      }
      if (!data) return;
      try {
        App.networkMap.data = data;
        App.networkMap.render(data);
        App.networkMap.populateModelFilter(data);
        App.networkMap.renderRecentRoutes(data.recent_routes);
        App.networkMap.renderPipelinePath();
      } catch (e) {}
    },

    render: function(data) {
      if (!data || !data.regions) return;
      var regions = data.regions;
      var filter = (document.getElementById('map-model-filter') || {}).value || '';

      var counts = {};
      var maxCount = 0;
      var totalNodes = 0;
      var totalRegions = 0;
      var codes = Object.keys(regions);
      for (var i = 0; i < codes.length; i++) {
        var code = codes[i];
        var r = regions[code];
        var count;
        if (filter && r.models) {
          count = r.models[filter] || 0;
        } else {
          count = r.total || 0;
        }
        if (count > 0) {
          counts[code] = count;
          totalNodes += count;
          if (App.networkMap.isPlaced(code)) totalRegions++;
          if (count > maxCount) maxCount = count;
        }
      }

      App.networkMap._applyRegionColors(counts, maxCount);

      // Pulsing dots, one per country with a computer in it, at the country's
      // marker (`regionCenter`). A country too small for the outlines has
      // only its dot, so its dot is what answers the pointer.
      var svg = document.querySelector('.world-svg');
      if (svg) {
        svg.querySelectorAll('.map-node-dot').forEach(function(d) { d.remove(); });
        var activeCodes = Object.keys(counts);
        for (var k = 0; k < activeCodes.length; k++) {
          var cc = activeCodes[k];
          var at = App.networkMap.regionCenter(cc);
          if (!at) continue;
          var alone = !App.networkMap.paths[cc];
          var dot = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
          dot.setAttribute('cx', at[0]);
          dot.setAttribute('cy', at[1]);
          dot.setAttribute('r', Math.max(3, Math.min(8, counts[cc] * 2)));
          dot.setAttribute('fill', 'rgba(59,130,246,0.7)');
          dot.setAttribute('class', 'map-node-dot' + (alone ? ' alone' : ''));
          if (alone) App.networkMap._hoverShows(dot, cc);
          svg.appendChild(dot);
        }
        App.networkMap.renderThisComputer(svg, data.self_region);
      }

      // Coverage health coloring
      for (var ci = 0; ci < codes.length; ci++) {
        var cc2 = codes[ci];
        var r2 = regions[cc2];
        var el2 = document.getElementById('region-' + cc2);
        if (!el2 || !counts[cc2]) continue;
        if (r2.coverage_gaps && r2.coverage_gaps.length > 0) {
          var gapRatio = r2.coverage_gaps.length / Math.max(Object.keys(r2.models || {}).length + r2.coverage_gaps.length, 1);
          if (gapRatio > 0.5) {
            el2.style.stroke = 'rgba(239,68,68,' + (0.5 + (counts[cc2] / Math.max(maxCount, 1)) * 0.5).toFixed(2) + ')';
          } else {
            el2.style.stroke = 'rgba(234,179,8,' + (0.5 + (counts[cc2] / Math.max(maxCount, 1)) * 0.5).toFixed(2) + ')';
          }
        }
      }

      // Regional health summary
      var totalGaps = 0;
      var topDemandModel = '';
      var topDemandRate = 0;
      for (var di = 0; di < codes.length; di++) {
        var rd = regions[codes[di]];
        if (rd.coverage_gaps) totalGaps += rd.coverage_gaps.length;
        if (rd.demand) {
          var dkeys = Object.keys(rd.demand);
          for (var dk = 0; dk < dkeys.length; dk++) {
            if (rd.demand[dkeys[dk]] > topDemandRate) {
              topDemandRate = rd.demand[dkeys[dk]];
              topDemandModel = dkeys[dk];
            }
          }
        }
      }

      App.networkMap._updateMapStats(totalNodes, totalRegions, maxCount);
      var statsEl = document.getElementById('map-stats-text');

      var healthEl = document.getElementById('map-regional-health');
      if (!healthEl) {
        healthEl = document.createElement('div');
        healthEl.id = 'map-regional-health';
        healthEl.className = 'text-xs text-muted mt-1';
        if (statsEl && statsEl.parentNode) statsEl.parentNode.appendChild(healthEl);
      }
      if (healthEl) {
        var healthText = I18n.t(totalRegions === 1 ? 'map.stats_region' : 'map.stats_regions', { count: totalRegions });
        if (totalGaps > 0) healthText += ' | ' + I18n.t(totalGaps === 1 ? 'map.stats_gaps' : 'map.stats_gaps_plural', { count: totalGaps });
        if (topDemandModel && topDemandRate > 0.1) {
          var shortName = topDemandModel.length > 20 ? topDemandModel.substring(0, 20) + '...' : topDemandModel;
          healthText += ' | ' + I18n.t('map.stats_top_demand', { model: shortName, rate: topDemandRate.toFixed(1) });
        }
        healthEl.textContent = healthText;
      }
    },

    applyFilter: function() {
      if (App.networkMap.data) App.networkMap.render(App.networkMap.data);
      App.networkMap.renderPipelinePath();
    },

    // WHERE A REGION SITS ON THE MAP, and the arc between two of them.
    //
    // Both the scheduler's PLANNED route and the routes that actually ran are
    // drawn from these, so a predicted hop and a real one land on the same
    // pixels — otherwise the map would show the plan bending one way and
    // history bending another between the same two countries.
    // "??" is a computer whose country is not known (`api::admin::map_region_of`):
    // counted among the computers, never as a region, and drawn nowhere.
    isPlaced: function(code) {
      return code !== '??';
    },

    // A country's marker: its label point (`points`, see above), never the
    // middle of its outline. Also needs no layout, so it is right while the
    // tab is hidden — `getBBox` there reads an unrendered element.
    regionCenter: function(code) {
      if (!code || !App.networkMap.isPlaced(code)) return null;
      var at = App.networkMap.points[code];
      return at ? [Math.round(at[0] * 10) / 10, Math.round(at[1] * 10) / 10] : null;
    },

    arcPath: function(a, b) {
      var mx = (a[0] + b[0]) / 2, my = (a[1] + b[1]) / 2;
      var dx = b[0] - a[0], dy = b[1] - a[1];
      var lift = Math.min(60, Math.sqrt(dx * dx + dy * dy) * 0.25);
      return 'M' + a[0] + ',' + a[1] + ' Q' + mx + ',' + (my - lift) + ' ' + b[0] + ',' + b[1];
    },

    // WORK THAT ACTUALLY CROSSED THE MAP.
    //
    // The planned path below answers "where would the next request go"; this
    // answers "where have requests been", which is what makes the map look
    // like a network rather than a list of pins. Newest first from the API, so
    // the index is the recency — the freshest arc is brightest and the oldest
    // is nearly gone.
    //
    // It draws nothing on a node that has served nothing across regions, and
    // that is the honest result: a new node's map is quiet because its swarm
    // has not asked it for anything yet.
    renderRecentRoutes: function(routes) {
      var svg = document.querySelector('.world-svg');
      if (!svg) return;
      svg.querySelectorAll('.map-route-history').forEach(function(el) { el.remove(); });
      // Written on EVERY call, including the empty one — the key is part of
      // drawing the arcs, so it cannot fall out of step with them.
      App.networkMap.renderRouteKey(routes);
      if (!routes || routes.length === 0) return;

      var ns = 'http://www.w3.org/2000/svg';
      var group = document.createElementNS(ns, 'g');
      group.setAttribute('class', 'map-route-history');

      routes.forEach(function(route, idx) {
        var pts = [];
        (route.regions || []).forEach(function(code) {
          var c = App.networkMap.regionCenter(code);
          if (c) pts.push(c);
        });
        if (pts.length < 2) return;
        // Freshest at full strength, fading to a trace by the end of the ring.
        var strength = Math.max(0.12, 1 - (idx / Math.max(routes.length, 1)) * 0.85);
        for (var i = 0; i < pts.length - 1; i++) {
          var path = document.createElementNS(ns, 'path');
          path.setAttribute('d', App.networkMap.arcPath(pts[i], pts[i + 1]));
          path.setAttribute('class', 'map-route-arc' + (route.ok ? '' : ' failed'));
          path.setAttribute('style', 'opacity:' + strength.toFixed(2));
          var t = document.createElementNS(ns, 'title');
          t.textContent = (route.model || '') + ' — ' + (route.regions || []).join(' \u2192 ');
          path.appendChild(t);
          group.appendChild(path);
        }
      });
      svg.appendChild(group);
    },

    // WHAT THE LINES MEAN, and what it means when there are none.
    //
    // The arcs are painted in two colours — a request that ran, and one that
    // failed — and were explained nowhere. A key must list every value the
    // thing it explains can paint (`docs/invariants/frontend.md`), and this one
    // listed only node density, which is the map's other visual channel.
    //
    // The empty case is the one that mattered enough to open FUTURE_WORK #62.
    // These are THIS node's own routes, so a node that has served nothing
    // across regions draws nothing — which is every new node, and a new user
    // opening the Network tab reads pins-and-no-traffic as "nothing is
    // happening here" rather than "nothing has happened *to you* yet". Saying
    // so is the honest half of that entry; seeding the map from the swarm's
    // traffic is the other half and is still open, because it needs a new
    // gossip message and a privacy decision. **Do not fake it with sample
    // data** — the arcs are worth something precisely because they are the
    // routes that actually ran.
    renderRouteKey: function(routes) {
      var el = document.getElementById('map-route-legend');
      if (!el) return;
      if (!routes || routes.length === 0) {
        el.innerHTML = '<span class="text-muted" style="font-size:0.75rem">'
          + U.escapeHtml(I18n.t('network.no_routes_yet')) + '</span>';
        return;
      }
      // Only offer the "failed" key when a failed arc is actually on the map:
      // a key that lists a colour nothing is painted in reads as a warning
      // about something that has not happened.
      var anyFailed = routes.some(function(r) { return !r.ok; });
      var items = [['', 'network.route_key_ran'], ['newest', 'network.route_key_newest']];
      if (anyFailed) items.push(['failed', 'network.route_key_failed']);
      el.innerHTML = items.map(function(it) {
        return '<span class="map-route-key-item">'
          + '<svg class="map-route-key-swatch" viewBox="0 0 22 8" aria-hidden="true">'
          + '<path d="M1,6 Q11,0 21,6" class="map-route-arc' + (it[0] === 'failed' ? ' failed' : '') + '"'
          + (it[0] === 'newest' ? '' : ' style="opacity:0.35"') + '/></svg>'
          + '<span class="text-muted" style="font-size:0.75rem">' + U.escapeHtml(I18n.t(it[1])) + '</span>'
          + '</span>';
      }).join('');
    },

    // Fetch the scheduler's plan for the currently-filtered model and draw
    // arcs between each chosen peer's region on the world map — the literal
    // inference path for the next request to that model.
    renderPipelinePath: function() {
      var svg = document.querySelector('.world-svg');
      if (!svg) return;
      svg.querySelectorAll('.map-pipeline-path').forEach(function(el) { el.remove(); });

      var filter = (document.getElementById('map-model-filter') || {}).value || '';
      if (!filter) return;

      App.data.loadPipelinePlan(filter)
        .then(function(plan) {
          if (!plan || !plan.segments || plan.segments.length === 0) return;
          var regionCenter = App.networkMap.regionCenter;
          var local = regionCenter(plan.local_region);
          var pts = [];
          if (local) pts.push({ c: local, label: I18n.t('map.pipeline_you'), local: true });
          plan.segments.forEach(function(seg, i) {
            if (seg.is_local && pts.length && pts[pts.length - 1].local) return;
            var c = regionCenter(seg.region);
            if (c) pts.push({
              c: c,
              label: String(i + 1),
              local: seg.is_local,
              nickname: seg.nickname,
              latency: seg.latency_ms,
            });
          });
          if (pts.length < 2) return;

          var ns = 'http://www.w3.org/2000/svg';
          var group = document.createElementNS(ns, 'g');
          group.setAttribute('class', 'map-pipeline-path');

          for (var i = 0; i < pts.length - 1; i++) {
            var path = document.createElementNS(ns, 'path');
            path.setAttribute('d', App.networkMap.arcPath(pts[i].c, pts[i + 1].c));
            path.setAttribute('class', 'map-pipeline-arc');
            group.appendChild(path);
          }

          pts.forEach(function(p, i) {
            var ring = document.createElementNS(ns, 'circle');
            ring.setAttribute('cx', p.c[0]);
            ring.setAttribute('cy', p.c[1]);
            ring.setAttribute('r', 8);
            ring.setAttribute('class', 'map-pipeline-node' + (p.local ? ' local' : ''));
            var title = (p.local ? I18n.t('map.pipeline_you') : I18n.t('map.pipeline_hop') + ' ' + p.label) +
              (p.nickname ? ' — ' + p.nickname : '') +
              (typeof p.latency === 'number' ? ' (' + p.latency + 'ms)' : '');
            var t = document.createElementNS(ns, 'title');
            t.textContent = title;
            ring.appendChild(t);
            group.appendChild(ring);
            var lbl = document.createElementNS(ns, 'text');
            lbl.setAttribute('x', p.c[0]);
            lbl.setAttribute('y', p.c[1] + 3);
            lbl.setAttribute('class', 'map-pipeline-label');
            lbl.textContent = p.label;
            group.appendChild(lbl);
          });

          svg.appendChild(group);
        })
        .catch(function() { /* quiet */ });
    },

    populateModelFilter: function(data) {
      var sel = document.getElementById('map-model-filter');
      if (!sel || !data || !data.regions) return;
      var models = {};
      var codes = Object.keys(data.regions);
      for (var i = 0; i < codes.length; i++) {
        var r = data.regions[codes[i]];
        if (r.models) {
          var mids = Object.keys(r.models);
          for (var j = 0; j < mids.length; j++) models[mids[j]] = true;
        }
      }
      var current = sel.value;
      sel.innerHTML = '<option value="">' + (typeof I18n !== 'undefined' ? I18n.t('dashboard.all_models') : 'All models') + '</option>';
      var sorted = Object.keys(models).sort();
      for (var k = 0; k < sorted.length; k++) {
        var opt = document.createElement('option');
        opt.value = sorted[k];
        opt.textContent = sorted[k].length > 30 ? sorted[k].substring(0, 30) + '...' : sorted[k];
        if (sorted[k] === current) opt.selected = true;
        sel.appendChild(opt);
      }
    },

    updateFromWs: function(regionSummary) {
      if (!App.networkMap.mapRendered) return;
      var maxCount = 0;
      var totalNodes = 0;
      var totalRegions = 0;
      var codes = Object.keys(regionSummary);
      for (var i = 0; i < codes.length; i++) {
        var count = regionSummary[codes[i]];
        if (count > 0) {
          totalNodes += count;
          if (App.networkMap.isPlaced(codes[i])) totalRegions++;
          if (count > maxCount) maxCount = count;
        }
      }
      App.networkMap._applyRegionColors(regionSummary, maxCount);
      App.networkMap._updateMapStats(totalNodes, totalRegions, maxCount);
    },

    /// Set the map header stats line + legend max — called from both
    /// `render()` (full topology rebuild) and `updateFromWs()` (live
    /// regionSummary refresh) so the formatting stays in one place.
    _updateMapStats: function(totalNodes, totalRegions, maxCount) {
      var statsEl = document.getElementById('map-stats-text');
      if (statsEl) {
        statsEl.textContent =
          I18n.t(totalNodes === 1 ? 'map.stats_nodes' : 'map.stats_nodes_plural', { count: totalNodes })
          + ' ' + I18n.t('map.stats_across') + ' '
          + I18n.t(totalRegions === 1 ? 'map.stats_region' : 'map.stats_regions', { count: totalRegions });
      }
      var legendEl = document.getElementById('map-legend-max');
      if (legendEl) legendEl.textContent = maxCount;
      App.networkMap._renderSwarmSummary();
    },

    /// Summarise what the swarm can actually DO, under the map.
    ///
    /// The map answers "where are the nodes"; on its own that leaves "so what
    /// can this network run?" unanswered, which is the question a new user
    /// actually has. Reuses the capacity snapshot the dashboard banner already
    /// renders (and its existing strings) rather than inventing a second
    /// vocabulary for the same numbers.
    _renderSwarmSummary: function() {
      var el = document.getElementById('map-swarm-summary');
      if (!el) return;
      var cap = App.data && App.data.cache && App.data.cache.stats
        ? App.data.cache.stats.swarm_capacity
        : null;
      if (!cap) { el.style.display = 'none'; return; }

      var nodes = cap.online_nodes || 0;
      var gpus = cap.gpu_nodes || 0;
      var vram = cap.total_vram_mb || 0;
      var disk = cap.total_disk_mb || 0;
      var serveable = (cap.serveable_models || []).length;

      var parts = [];
      parts.push('<strong>' + nodes + '</strong> ' +
        U.escapeHtml(I18n.t(nodes === 1 ? 'netstatus.res_computers_one' : 'netstatus.res_computers_other')));
      parts.push('<strong>' + gpus + '</strong> ' +
        U.escapeHtml(I18n.t(gpus === 1 ? 'netstatus.res_gpu_one' : 'netstatus.res_gpu_other')));
      if (vram > 0) {
        parts.push('<strong>' + U.escapeHtml(U.formatMB(vram)) + '</strong> ' +
          U.escapeHtml(I18n.t('pool.combined_vram')));
      }
      if (disk > 0) {
        parts.push('<strong>' + U.escapeHtml(U.formatMB(disk)) + '</strong> ' +
          U.escapeHtml(I18n.t('netstatus.res_disk')));
      }
      el.innerHTML = parts.join(' \u00b7 ');
      el.style.display = '';

      // Which models the swarm can actually serve right now — the payoff line.
      var modelsEl = document.getElementById('map-swarm-models');
      if (!modelsEl) return;
      if (serveable === 0) { modelsEl.style.display = 'none'; return; }
      // Named from the id, as the dashboard banner names them: the file's own
      // name is shared by two quantisations of one model.
      var names = (cap.serveable_models || []).slice(0, 4).map(function (m) {
        return U.escapeHtml(U.formatModelDisplayName(m.model_id || m.display_name || ''));
      }).join(', ');
      var more = serveable > 4 ? ' +' + (serveable - 4) : '';
      modelsEl.innerHTML = U.escapeHtml(I18n.t('netstatus.runs')) + ' ' + names + more;
      modelsEl.style.display = '';
    },

    // Per-locale display name for an ISO 3166-1 alpha-2 code. Intl.DisplayNames
    // is Chrome 81+/FF 86+/Safari 14.1+ — covers everything we ship to. Falls
    // back to the raw code on the very-old-browser path.
    _displayNamesCache: {},
    countryName: function(code) {
      var lang = (window.I18n && I18n.getLang && I18n.getLang()) || 'en';
      var cache = App.networkMap._displayNamesCache;
      if (!cache[lang]) {
        try { cache[lang] = new Intl.DisplayNames([lang], { type: 'region' }); }
        catch (e) { cache[lang] = null; }
      }
      try { return (cache[lang] && cache[lang].of(code)) || code; }
      catch (e) { return code; }
    },

    showTooltip: function(event, code) {
      App.networkMap.hideTooltip();
      var info = App.networkMap.data && App.networkMap.data.regions ? App.networkMap.data.regions[code] : null;
      var tip = document.createElement('div');
      tip.id = 'map-tooltip';
      tip.className = 'map-tooltip';
      var countryName = App.networkMap.countryName(code);
      var html = '<strong>' + countryName + '</strong> <span class="text-muted" style="font-size:0.7rem">' + code + '</span>';
      if (info) {
        html += '<span class="mono" style="margin-left:8px">' + U.escapeHtml(I18n.t(info.total === 1 ? 'map.stats_nodes' : 'map.stats_nodes_plural', { count: info.total })) + '</span>';
        if (info.models) {
          var mids = Object.keys(info.models);
          if (mids.length > 0) {
            html += '<div class="mt-1" style="font-size:0.75rem">';
            for (var i = 0; i < Math.min(mids.length, 5); i++) {
              var mName = U.formatModelDisplayName(mids[i]);
              if (mName.length > 22) mName = mName.substring(0, 22) + '...';
              var demandStr = '';
              if (info.demand && info.demand[mids[i]]) {
                demandStr = ' <span style="color:var(--accent)">' + I18n.t('map.tooltip_demand_rate', { rate: info.demand[mids[i]].toFixed(1) }) + '</span>';
              }
              html += '<div class="flex-between" style="gap:12px"><span class="text-muted">' + U.escapeHtml(mName) + '</span><span class="mono">' + U.escapeHtml(String(info.models[mids[i]])) + demandStr + '</span></div>';
            }
            if (mids.length > 5) html += '<div class="text-muted">' + U.escapeHtml(I18n.t('map.tooltip_more', { count: mids.length - 5 })) + '</div>';
            html += '</div>';
          }
        }
        if (info.coverage_gaps && info.coverage_gaps.length > 0) {
          html += '<div class="mt-1" style="font-size:0.7rem;color:var(--yellow)">' + I18n.t('map.coverage_gaps') + ': ' + U.escapeHtml(I18n.t(info.coverage_gaps.length === 1 ? 'map.tooltip_models' : 'map.tooltip_models_plural', { count: info.coverage_gaps.length })) + '</div>';
        }
      } else {
        html += '<span class="text-muted" style="margin-left:8px">' + U.escapeHtml(I18n.t('map.tooltip_no_nodes')) + '</span>';
      }
      tip.innerHTML = html;
      var mapContainer = document.getElementById('world-map-container');
      mapContainer.appendChild(tip);
      App.networkMap.moveTooltip(event);
      requestAnimationFrame(function() { tip.classList.add('visible'); });
    },

    moveTooltip: function(event) {
      var tip = document.getElementById('map-tooltip');
      if (!tip) return;
      var mapContainer = document.getElementById('world-map-container');
      var containerRect = mapContainer.getBoundingClientRect();
      var x = event.clientX - containerRect.left + 14;
      var y = event.clientY - containerRect.top - tip.offsetHeight - 10;
      if (x + tip.offsetWidth > containerRect.width - 8) x = event.clientX - containerRect.left - tip.offsetWidth - 14;
      if (y < 4) y = event.clientY - containerRect.top + 18;
      tip.style.left = x + 'px';
      tip.style.top = y + 'px';
    },

    hideTooltip: function() {
      var tip = document.getElementById('map-tooltip');
      if (tip) tip.remove();
    }
  };
})();
