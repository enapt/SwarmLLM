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
          var intensity = Math.max(0.25, n / Math.max(maxCount, 1));
          var fillAlpha = (0.06 + intensity * 0.14).toFixed(2);
          var strokeAlpha = (0.5 + intensity * 0.5).toFixed(2);
          el.style.fill = 'rgba(59,130,246,' + fillAlpha + ')';
          el.style.stroke = 'rgba(100,180,255,' + strokeAlpha + ')';
          el.style.strokeWidth = (0.8 + intensity * 1.2).toFixed(1);
          el.setAttribute('filter', 'url(#glow-md)');
        }
      }
    },

    numToAlpha2: {
      '004':'AF','008':'AL','012':'DZ','024':'AO','032':'AR','036':'AU','040':'AT',
      '044':'BS','050':'BD','056':'BE','064':'BT','068':'BO','070':'BA','072':'BW',
      '076':'BR','084':'BZ','090':'SB','096':'BN','100':'BG','104':'MM','108':'BI',
      '112':'BY','116':'KH','120':'CM','124':'CA','140':'CF','144':'LK','148':'TD',
      '152':'CL','156':'CN','158':'TW','170':'CO','178':'CG','180':'CD','188':'CR',
      '191':'HR','192':'CU','196':'CY','203':'CZ','204':'BJ','208':'DK','214':'DO',
      '218':'EC','222':'SV','226':'GQ','231':'ET','232':'ER','233':'EE','238':'FK',
      '242':'FJ','246':'FI','250':'FR','260':'TF','262':'DJ','266':'GA','268':'GE',
      '270':'GM','275':'PS','276':'DE','288':'GH','296':'KI','300':'GR','304':'GL',
      '320':'GT','324':'GN','328':'GY','332':'HT','340':'HN','344':'HK','348':'HU',
      '352':'IS','356':'IN','360':'ID','364':'IR','368':'IQ','372':'IE','376':'IL',
      '380':'IT','384':'CI','388':'JM','392':'JP','398':'KZ','400':'JO','404':'KE',
      '408':'KP','410':'KR','414':'KW','417':'KG','418':'LA','422':'LB','426':'LS',
      '428':'LV','430':'LR','434':'LY','440':'LT','442':'LU','450':'MG','454':'MW',
      '458':'MY','462':'MV','466':'ML','478':'MR','484':'MX','496':'MN','498':'MD',
      '504':'MA','508':'MZ','512':'OM','516':'NA','524':'NP','528':'NL','540':'NC',
      '548':'VU','554':'NZ','558':'NI','562':'NE','566':'NG','578':'NO','586':'PK',
      '591':'PA','598':'PG','600':'PY','604':'PE','608':'PH','616':'PL','620':'PT',
      '624':'GW','626':'TL','634':'QA','642':'RO','643':'RU','646':'RW','682':'SA',
      '686':'SN','694':'SL','702':'SG','703':'SK','704':'VN','706':'SO','710':'ZA',
      '716':'ZW','724':'ES','728':'SS','729':'SD','732':'EH','740':'SR','752':'SE',
      '756':'CH','760':'SY','762':'TJ','764':'TH','768':'TG','780':'TT','784':'AE',
      '788':'TN','792':'TR','795':'TM','800':'UG','804':'UA','818':'EG','826':'GB',
      '834':'TZ','840':'US','854':'BF','858':'UY','860':'UZ','862':'VE','887':'YE',
      '894':'ZM',
    },

    projectCoord: function(lon, lat) {
      var x = (lon + 180) / 360 * 1000;
      var y = (90 - lat) / 180 * 500;
      return [Math.round(x * 10) / 10, Math.round(y * 10) / 10];
    },

    ringToPath: function(ring) {
      var parts = [];
      for (var i = 0; i < ring.length; i++) {
        var p = App.networkMap.projectCoord(ring[i][0], ring[i][1]);
        if (i === 0) {
          parts.push('M' + p[0] + ',' + p[1]);
        } else {
          var lonDiff = Math.abs(ring[i][0] - ring[i - 1][0]);
          if (lonDiff > 180) {
            parts.push('M' + p[0] + ',' + p[1]);
          } else {
            parts.push('L' + p[0] + ',' + p[1]);
          }
        }
      }
      parts.push('Z');
      return parts.join('');
    },

    geomToPath: function(geom) {
      var d = '';
      if (geom.type === 'Polygon') {
        for (var i = 0; i < geom.coordinates.length; i++) {
          d += App.networkMap.ringToPath(geom.coordinates[i]);
        }
      } else if (geom.type === 'MultiPolygon') {
        for (var i = 0; i < geom.coordinates.length; i++) {
          for (var j = 0; j < geom.coordinates[i].length; j++) {
            d += App.networkMap.ringToPath(geom.coordinates[i][j]);
          }
        }
      }
      return d;
    },

    paths: {},

    buildSvg: async function() {
      var container = document.getElementById('world-map');
      if (!container) return;

      try {
        var resp = await fetch('/static/data/countries-110m.json');
        var topo = await resp.json();
        var geojson = topojson.feature(topo, topo.objects.countries);
        var features = geojson.features;

        App.networkMap.paths = {};
        for (var i = 0; i < features.length; i++) {
          var f = features[i];
          var numId = String(f.id);
          var alpha2 = App.networkMap.numToAlpha2[numId];
          if (!alpha2) continue;
          var d = App.networkMap.geomToPath(f.geometry);
          App.networkMap.paths[alpha2] = d;
        }
      } catch (e) {
      }

      var svg = '<svg viewBox="0 0 1000 500" xmlns="http://www.w3.org/2000/svg" class="world-svg">';
      svg += '<defs>';
      svg += '<filter id="glow-sm"><feGaussianBlur stdDeviation="1.5" result="blur"/><feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge></filter>';
      svg += '<filter id="glow-md"><feGaussianBlur stdDeviation="3" result="blur"/><feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge></filter>';
      svg += '</defs>';
      svg += '<rect width="1000" height="500" fill="transparent" rx="4"/>';
      for (var x = 0; x <= 1000; x += 50) {
        var op = (x % 100 === 0) ? '0.25' : '0.1';
        svg += '<line x1="' + x + '" y1="0" x2="' + x + '" y2="500" stroke="var(--accent)" stroke-width="0.3" opacity="' + op + '"/>';
      }
      for (var y = 0; y <= 500; y += 50) {
        var opy = (y % 100 === 0) ? '0.25' : '0.1';
        svg += '<line x1="0" y1="' + y + '" x2="1000" y2="' + y + '" stroke="var(--accent)" stroke-width="0.3" opacity="' + opy + '"/>';
      }
      svg += '<line x1="0" y1="250" x2="1000" y2="250" stroke="var(--accent)" stroke-width="0.5" opacity="0.15" stroke-dasharray="8,4"/>';
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
        el.addEventListener('mouseenter', function(e) { App.networkMap.showTooltip(e, el.dataset.code); });
        el.addEventListener('mousemove', function(e) { App.networkMap.moveTooltip(e); });
        el.addEventListener('mouseleave', function() { App.networkMap.hideTooltip(); });
      });

      App.networkMap.mapRendered = true;
    },

    refresh: async function() {
      if (!App.networkMap.mapRendered) await App.networkMap.buildSvg();
      try {
        var resp = await App.authFetch('/api/admin/network-map');
        var data = await resp.json();
        App.networkMap.data = data;
        App.networkMap.render(data);
        App.networkMap.populateModelFilter(data);
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
          totalRegions++;
          if (count > maxCount) maxCount = count;
        }
      }

      App.networkMap._applyRegionColors(counts, maxCount);

      // Pulsing dots
      var svg = document.querySelector('.world-svg');
      if (svg) {
        svg.querySelectorAll('.map-node-dot').forEach(function(d) { d.remove(); });
        var activeCodes = Object.keys(counts);
        for (var k = 0; k < activeCodes.length; k++) {
          var cc = activeCodes[k];
          var regionEl = document.getElementById('region-' + cc);
          if (!regionEl) continue;
          var bbox = regionEl.getBBox();
          var cx = bbox.x + bbox.width / 2;
          var cy = bbox.y + bbox.height / 2;
          var dotR = Math.max(3, Math.min(8, counts[cc] * 2));
          var dot = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
          dot.setAttribute('cx', cx);
          dot.setAttribute('cy', cy);
          dot.setAttribute('r', dotR);
          dot.setAttribute('fill', 'rgba(59,130,246,0.7)');
          dot.setAttribute('class', 'map-node-dot');
          svg.appendChild(dot);
        }
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

      var statsEl = document.getElementById('map-stats-text');
      var statsText = I18n.t(totalNodes === 1 ? 'map.stats_nodes' : 'map.stats_nodes_plural', { count: totalNodes }) + ' ' + I18n.t('map.stats_across') + ' ' + I18n.t(totalRegions === 1 ? 'map.stats_region' : 'map.stats_regions', { count: totalRegions });
      if (statsEl) statsEl.textContent = statsText;
      document.getElementById('map-legend-max').textContent = maxCount;

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
          totalRegions++;
          if (count > maxCount) maxCount = count;
        }
      }
      App.networkMap._applyRegionColors(regionSummary, maxCount);
      var statsEl = document.getElementById('map-stats-text');
      if (statsEl) statsEl.textContent = I18n.t(totalNodes === 1 ? 'map.stats_nodes' : 'map.stats_nodes_plural', { count: totalNodes }) + ' ' + I18n.t('map.stats_across') + ' ' + I18n.t(totalRegions === 1 ? 'map.stats_region' : 'map.stats_regions', { count: totalRegions });
      document.getElementById('map-legend-max').textContent = maxCount;
    },

    countryNames: {US:'United States',CA:'Canada',MX:'Mexico',BR:'Brazil',AR:'Argentina',CL:'Chile',CO:'Colombia',GB:'United Kingdom',FR:'France',DE:'Germany',ES:'Spain',IT:'Italy',NL:'Netherlands',SE:'Sweden',NO:'Norway',FI:'Finland',PL:'Poland',UA:'Ukraine',RU:'Russia',TR:'Turkey',IN:'India',CN:'China',JP:'Japan',KR:'South Korea',AU:'Australia',NZ:'New Zealand',ZA:'South Africa',NG:'Nigeria',EG:'Egypt',KE:'Kenya',SG:'Singapore',ID:'Indonesia',TH:'Thailand',VN:'Vietnam',PH:'Philippines',TW:'Taiwan',IL:'Israel',AE:'UAE',SA:'Saudi Arabia',CH:'Switzerland',AT:'Austria',CZ:'Czech Republic',RO:'Romania',IE:'Ireland',PT:'Portugal',DK:'Denmark',BE:'Belgium'},

    showTooltip: function(event, code) {
      App.networkMap.hideTooltip();
      var info = App.networkMap.data && App.networkMap.data.regions ? App.networkMap.data.regions[code] : null;
      var tip = document.createElement('div');
      tip.id = 'map-tooltip';
      tip.className = 'map-tooltip';
      var countryName = App.networkMap.countryNames[code] || code;
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
