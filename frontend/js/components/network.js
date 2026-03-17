'use strict';

// ============================================================================
// SwarmLLM — Network Component
// Network Map (SVG world heatmap), Identity/Leaderboard, Network Code, Compare
// ============================================================================

(function() {
  var S = App.state;
  var U = App.utils;

  // ========================================================================
  // Identity / Leaderboard
  // ========================================================================
  App.identity = {
    loadNickname: async function() {
      try {
        var resp = await fetch('/api/identity/nickname');
        if (!resp.ok) return;
        var data = await resp.json();
        var nickEl = document.getElementById('settings-nickname');
        var visEl = document.getElementById('settings-visibility');
        if (nickEl && data.nickname) nickEl.value = data.nickname;
        if (visEl && data.visibility) visEl.value = data.visibility;
      } catch (e) {}
    },

    saveNickname: async function() {
      var nickEl = document.getElementById('settings-nickname');
      var visEl = document.getElementById('settings-visibility');
      if (!nickEl) return;
      var nickname = nickEl.value.trim();

      if (!nickname) {
        try {
          await App.authFetch('/api/identity/nickname', { method: 'DELETE' });
        } catch (e) {}
        return;
      }

      try {
        var resp = await App.authFetch('/api/identity/nickname', {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            nickname: nickname,
            visibility: visEl ? visEl.value : 'nickname',
          }),
        });
        if (!resp.ok) {
          var err = await resp.json().catch(function() { return {}; });
          App.ui.showBanner('error', err.error ? err.error.message : 'Failed to set nickname');
        }
      } catch (e) {
        App.ui.showBanner('error', 'Error saving nickname: ' + e.message);
      }
    },

    loadLeaderboard: async function() {
      var tbody = document.getElementById('leaderboard-body');
      if (!tbody) return;

      try {
        var resp = await fetch('/api/identity/leaderboard?limit=50');
        if (!resp.ok) { tbody.innerHTML = '<tr><td colspan="4" class="text-muted" style="text-align:center">Failed to load</td></tr>'; return; }
        var data = await resp.json();
        var entries = data.leaderboard || [];

        if (entries.length === 0) {
          tbody.innerHTML = '<tr><td colspan="4" class="text-muted" style="text-align:center;padding:24px">No activity yet. Credits are earned by helping others run AI models.</td></tr>';
          return;
        }

        var html = '';
        for (var i = 0; i < entries.length; i++) {
          var e = entries[i];
          var tierClass = (e.tier || 'silver').toLowerCase().replace(/[^a-z]/g, '');
          html += '<tr>'
            + '<td class="mono">' + (e.rank || i+1) + '</td>'
            + '<td>' + (e.display_name !== e.node_id ? U.escapeHtml(e.display_name) + ' <span class="text-muted mono" style="font-size:0.75rem">' + U.escapeHtml(e.node_id) + '</span>' : '<span class="mono">' + U.escapeHtml(e.node_id) + '</span>') + '</td>'
            + '<td class="mono">' + (e.credits || 0) + '</td>'
            + '<td><span class="tier-badge ' + tierClass + '">' + U.escapeHtml(e.tier || 'Silver') + '</span></td>'
            + '</tr>';
        }
        tbody.innerHTML = html;
      } catch (e) {
        tbody.innerHTML = '<tr><td colspan="4" class="text-muted" style="text-align:center">Error: ' + U.escapeHtml(e.message) + '</td></tr>';
      }
    }
  };

  // ========================================================================
  // Network Code
  // ========================================================================
  App.networkCode = {
    load: async function() {
      try {
        var resp = await App.authFetch('/api/admin/network-code');
        var data = await resp.json();
        var codeInput = document.getElementById('my-network-code');
        if (codeInput && data.code) codeInput.value = data.code;
      } catch (e) {}
    },

    copy: function() {
      var input = document.getElementById('my-network-code');
      var btn = document.getElementById('btn-copy-network-code');
      if (input && input.value) {
        navigator.clipboard.writeText(input.value).then(function() {
          if (btn) { btn.textContent = 'Copied!'; btn.style.color = 'var(--green)'; setTimeout(function() { btn.textContent = 'Copy'; btn.style.color = ''; }, 2000); }
          App.notifications.showToast('Network code copied to clipboard', 'success');
        }).catch(function() {
          App.ui.showBanner('error', 'Failed to copy \u2014 try selecting and copying manually');
        });
      }
    },

    join: async function() {
      var input = document.getElementById('join-code-input');
      var status = document.getElementById('join-status');
      if (!input || !input.value.trim()) return;

      if (status) { status.textContent = 'Connecting...'; status.style.color = 'var(--text-muted)'; }

      try {
        var resp = await App.authFetch('/api/admin/join-network', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ code: input.value.trim() })
        });
        var data = await resp.json();
        if (resp.ok) {
          if (status) { status.textContent = 'Connected! Peer added.'; status.style.color = 'var(--green)'; }
          input.value = '';
          App.notifications.showToast('Peer connected successfully', 'success');
          setTimeout(function() { App.networkCode.load(); }, 2000);
        } else {
          if (status) { status.textContent = data.error || 'Failed to join'; status.style.color = 'var(--red, #ff6464)'; }
        }
      } catch (e) {
        if (status) { status.textContent = 'Network error'; status.style.color = 'var(--red, #ff6464)'; }
      }
    }
  };

  // ========================================================================
  // Network Map — SVG world heatmap
  // ========================================================================
  App.networkMap = {
    data: null,
    mapRendered: false,

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
        console.warn('[SwarmLLM] Failed to load map data:', e.message);
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
        var resp = await fetch('/api/admin/network-map');
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

      var allCodes = Object.keys(App.networkMap.paths);
      for (var j = 0; j < allCodes.length; j++) {
        var c = allCodes[j];
        var el = document.getElementById('region-' + c);
        if (!el) continue;
        var n = counts[c] || 0;
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
      var statsText = totalNodes + (totalNodes === 1 ? ' node' : ' nodes') + ' across ' + totalRegions + (totalRegions === 1 ? ' region' : ' regions');
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
        var healthText = totalRegions + (totalRegions === 1 ? ' region' : ' regions');
        if (totalGaps > 0) healthText += ' | ' + totalGaps + ' coverage gap' + (totalGaps !== 1 ? 's' : '');
        if (topDemandModel && topDemandRate > 0.1) {
          var shortName = topDemandModel.length > 20 ? topDemandModel.substring(0, 20) + '...' : topDemandModel;
          healthText += ' | top demand: ' + shortName + ' (' + topDemandRate.toFixed(1) + ' req/10m)';
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
      sel.innerHTML = '<option value="">All models</option>';
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
      var allCodes = Object.keys(App.networkMap.paths);
      for (var j = 0; j < allCodes.length; j++) {
        var c = allCodes[j];
        var el = document.getElementById('region-' + c);
        if (!el) continue;
        var n = regionSummary[c] || 0;
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
      var statsEl = document.getElementById('map-stats-text');
      if (statsEl) statsEl.textContent = totalNodes + (totalNodes === 1 ? ' node' : ' nodes') + ' across ' + totalRegions + (totalRegions === 1 ? ' region' : ' regions');
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
        html += '<span class="mono" style="margin-left:8px">' + info.total + ' node' + (info.total !== 1 ? 's' : '') + '</span>';
        if (info.models) {
          var mids = Object.keys(info.models);
          if (mids.length > 0) {
            html += '<div class="mt-1" style="font-size:0.75rem">';
            for (var i = 0; i < Math.min(mids.length, 5); i++) {
              var mName = U.formatModelDisplayName(mids[i]);
              if (mName.length > 22) mName = mName.substring(0, 22) + '...';
              var demandStr = '';
              if (info.demand && info.demand[mids[i]]) {
                demandStr = ' <span style="color:var(--color-accent)">' + info.demand[mids[i]].toFixed(1) + ' req/10m</span>';
              }
              html += '<div class="flex-between" style="gap:12px"><span class="text-muted">' + U.escapeHtml(mName) + '</span><span class="mono">' + U.escapeHtml(String(info.models[mids[i]])) + demandStr + '</span></div>';
            }
            if (mids.length > 5) html += '<div class="text-muted">+' + (mids.length - 5) + ' more</div>';
            html += '</div>';
          }
        }
        if (info.coverage_gaps && info.coverage_gaps.length > 0) {
          html += '<div class="mt-1" style="font-size:0.7rem;color:var(--color-warning)">' + I18n.t('map.coverage_gaps') + ': ' + info.coverage_gaps.length + ' model' + (info.coverage_gaps.length !== 1 ? 's' : '') + '</div>';
        }
      } else {
        html += '<span class="text-muted" style="margin-left:8px">No nodes</span>';
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

  // ========================================================================
  // Model Compare
  // ========================================================================
  App.compare = {
    models: [],
    selected: [],
    running: false,

    loadModels: async function() {
      try {
        var container = document.getElementById('compare-model-list');
        if (!container) return;

        var localModels = [];
        var cloudModels = [];
        try {
          var resp = await App.authFetch('/api/admin/models');
          if (resp.ok) {
            var d = await resp.json();
            localModels = Array.isArray(d) ? d : (d.models || d.data || []);
          }
        } catch(e) {}
        try {
          var resp2 = await App.authFetch('/api/admin/provider-models');
          if (resp2.ok) {
            var d2 = await resp2.json();
            cloudModels = Array.isArray(d2) ? d2 : (d2.models || d2.data || []);
          }
        } catch(e) {}

        App.compare.models = [];
        (localModels || []).forEach(function(m) {
          App.compare.models.push({ id: m.id || m.model_id || m.name, type: 'local' });
        });
        (cloudModels || []).forEach(function(m) {
          var mid = m.id || m.model_id || m.name;
          if (!App.compare.models.some(function(x) { return x.id === mid; })) {
            var ctx = m.context_length || m.context_window || m.max_model_len || 0;
            App.compare.models.push({ id: mid, type: 'cloud', context: ctx });
          }
        });

        if (App.compare.models.length === 0) {
          container.innerHTML = '<span class="text-muted" style="font-size:0.8rem">No models available yet. Download a model or add a cloud provider in Settings first.</span>';
          return;
        }

        container.innerHTML = '';
        App.compare.models.forEach(function(m, idx) {
          var chip = document.createElement('label');
          chip.className = 'compare-model-chip type-' + m.type;
          chip.style.animationDelay = (idx * 30) + 'ms';
          var displayName = m.id.length > 35 ? m.id.substring(0, 35) + '...' : m.id;
          var ctxLabel = m.context && m.context > 0 ? ' \u00B7 ' + Math.round(m.context / 1000) + 'k ctx' : '';
          chip.innerHTML = '<input type="checkbox" value="' + U.escapeHtml(m.id) + '">' +
            '<span>' + U.escapeHtml(displayName) + '</span>' +
            '<span class="chip-type">' + m.type + ctxLabel + '</span>';
          chip.title = m.id + (ctxLabel ? ' (' + m.context + ' tokens)' : '');
          chip.querySelector('input').addEventListener('change', function() {
            chip.classList.toggle('selected', this.checked);
            App.compare.updateSelected();
          });
          container.appendChild(chip);
        });

        var filters = document.getElementById('compare-filters');
        if (filters) {
          filters.querySelectorAll('.compare-filter').forEach(function(btn) {
            btn.addEventListener('click', function() {
              filters.querySelectorAll('.compare-filter').forEach(function(b) { b.classList.remove('active'); });
              btn.classList.add('active');
              var f = btn.getAttribute('data-filter');
              container.querySelectorAll('.compare-model-chip').forEach(function(chip) {
                if (f === 'all') { chip.style.display = ''; }
                else { chip.style.display = chip.classList.contains('type-' + f) ? '' : 'none'; }
              });
            });
          });
        }
      } catch(e) {}
    },

    updateSelected: function() {
      App.compare.selected = [];
      var checks = document.querySelectorAll('#compare-model-list input[type="checkbox"]:checked');
      checks.forEach(function(cb) { App.compare.selected.push(cb.value); });
    },

    run: async function() {
      if (App.compare.running) return;
      var prompt = (document.getElementById('compare-prompt') || {}).value;
      if (!prompt || !prompt.trim()) {
        App.notifications.showToast('Enter a prompt to compare', 'error');
        return;
      }
      if (App.compare.selected.length < 2) {
        App.notifications.showToast('Select at least 2 models to compare', 'error');
        return;
      }
      if (App.compare.selected.length > 10) {
        App.notifications.showToast('Maximum 10 models per comparison', 'error');
        return;
      }

      var system = (document.getElementById('compare-system') || {}).value || '';
      var temperature = parseFloat((document.getElementById('compare-temp') || {}).value) || 0.7;
      var maxTokens = parseInt((document.getElementById('compare-max-tokens') || {}).value) || 1024;

      App.compare.running = true;
      var btn = document.getElementById('btn-compare-run');
      if (btn) { btn.disabled = true; btn.textContent = 'Running...'; }

      var resultsDiv = document.getElementById('compare-results');
      var n = App.compare.selected.length;
      var colClass = n <= 2 ? 'cols-2' : n <= 3 ? 'cols-3' : n <= 4 ? 'cols-4' : 'cols-many';
      resultsDiv.className = 'compare-results ' + colClass;

      resultsDiv.innerHTML = '';
      App.compare.selected.forEach(function(modelId) {
        var card = document.createElement('div');
        card.className = 'compare-card';
        card.id = 'compare-card-' + modelId.replace(/[^a-zA-Z0-9_-]/g, '_');
        card.innerHTML =
          '<div class="compare-card-header">' +
            '<span class="compare-card-model">' + U.escapeHtml(modelId) + '</span>' +
            '<span class="compare-card-meta"><span class="spinner" style="width:14px;height:14px"></span></span>' +
          '</div>' +
          '<div class="compare-card-body"><div class="compare-spinner"><div class="spinner"></div> Waiting for response...</div></div>';
        resultsDiv.appendChild(card);
      });

      var statusDiv = document.getElementById('compare-status');
      if (statusDiv) { statusDiv.style.display = ''; statusDiv.innerHTML = '<span class="text-muted">Sending prompt to ' + n + ' models concurrently...</span>'; }

      var promises = App.compare.selected.map(function(modelId) {
        var body = {
          model: modelId,
          max_tokens: maxTokens,
          temperature: temperature,
          messages: [{ role: 'user', content: prompt.trim() }],
          stream: false,
        };
        if (system.trim()) body.system = system.trim();

        var start = performance.now();
        var controller = new AbortController();
        var timeoutId = setTimeout(function() { controller.abort(); }, 45000);
        return App.authFetch('/v1/messages', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body),
          signal: controller.signal,
        }).then(function(resp) {
          clearTimeout(timeoutId);
          var elapsed = Math.round(performance.now() - start);
          return resp.json().then(function(data) {
            return { model: modelId, data: data, ok: resp.ok, latency_ms: elapsed };
          });
        }).catch(function(err) {
          clearTimeout(timeoutId);
          var msg = err.name === 'AbortError' ? 'Timed out after 45s' : err.message;
          return { model: modelId, error: msg, ok: false, latency_ms: Math.round(performance.now() - start) };
        });
      });

      var completed = 0;
      promises.forEach(function(p) {
        p.then(function(result) {
          completed++;
          App.compare.renderCard(result);
          if (statusDiv) {
            statusDiv.innerHTML = '<span class="text-muted">' + completed + ' / ' + n + ' models complete</span>';
            if (completed === n) {
              statusDiv.innerHTML = '<span style="color:var(--green)">All ' + n + ' models complete</span>';
              setTimeout(function() { statusDiv.style.display = 'none'; }, 3000);
            }
          }
        });
      });

      Promise.all(promises).then(function(results) {
        App.compare.running = false;
        if (btn) { btn.disabled = false; btn.textContent = 'Run Compare'; }
        try {
          var history = JSON.parse(localStorage.getItem('swarmllm_compare_history') || '[]');
          history.unshift({
            prompt: prompt.trim().substring(0, 200),
            models: App.compare.selected.slice(),
            timestamp: Date.now(),
            results: results.map(function(r) {
              var content = '';
              if (!r.error && r.ok) {
                (r.data.content || []).forEach(function(b) { if (b.type === 'text') content += b.text; });
              }
              return {
                model: r.model, ok: r.ok, error: r.error || null,
                latency_ms: r.latency_ms, content: content,
                input_tokens: r.ok ? ((r.data.usage || {}).input_tokens || 0) : 0,
                output_tokens: r.ok ? ((r.data.usage || {}).output_tokens || 0) : 0,
              };
            }),
          });
          if (history.length > 20) history = history.slice(0, 20);
          localStorage.setItem('swarmllm_compare_history', JSON.stringify(history));
          App.compare.renderHistory();
        } catch (e) {}
      });
    },

    renderHistory: function() {
      var container = document.getElementById('compare-history');
      if (!container) return;
      try {
        var history = JSON.parse(localStorage.getItem('swarmllm_compare_history') || '[]');
        if (history.length === 0) { container.style.display = 'none'; return; }
        container.style.display = '';
        var html = '<div style="font-size:0.75rem;color:var(--text-muted);margin-bottom:8px;text-transform:uppercase;letter-spacing:0.06em">Recent Comparisons</div>';
        history.slice(0, 10).forEach(function(item, idx) {
          var ago = App.compare.timeAgo(item.timestamp);
          var modelList = (item.models || []).map(function(m) {
            return m.split('/').pop().replace(/-\d{4}-\d{2}-\d{2}$/, '');
          }).join(', ');
          html += '<div class="compare-history-item" data-compare-idx="' + idx + '">' +
            '<span class="compare-history-prompt">' + U.escapeHtml(item.prompt) + '</span>' +
            '<span class="compare-history-meta">' + U.escapeHtml(modelList) + ' &middot; ' + ago + '</span>' +
          '</div>';
        });
        container.innerHTML = html;
      } catch (e) { container.style.display = 'none'; }
    },

    restoreFromHistory: function(item) {
      var promptEl = document.getElementById('compare-prompt');
      if (promptEl) promptEl.value = item.prompt;

      var resultsDiv = document.getElementById('compare-results');
      if (!resultsDiv || !item.results || !item.results.length) return;

      resultsDiv.innerHTML = '';
      item.results.forEach(function(r) {
        var card = document.createElement('div');
        card.className = 'compare-card';
        card.id = 'compare-card-' + r.model.replace(/[^a-zA-Z0-9_-]/g, '_');
        card.innerHTML = '<div class="compare-card-body"></div>';
        resultsDiv.appendChild(card);
        App.compare.renderCard({
          model: r.model, ok: r.ok, error: r.error,
          latency_ms: r.latency_ms,
          data: {
            content: [{ type: 'text', text: r.content || '' }],
            usage: { input_tokens: r.input_tokens, output_tokens: r.output_tokens },
          },
        });
      });

      var statusDiv = document.getElementById('compare-status');
      if (statusDiv) { statusDiv.style.display = ''; statusDiv.innerHTML = '<span class="text-muted">Restored from history &middot; ' + App.compare.timeAgo(item.timestamp) + '</span>'; }
    },

    timeAgo: function(ts) {
      var s = Math.floor((Date.now() - ts) / 1000);
      if (s < 60) return 'just now';
      if (s < 3600) return Math.floor(s / 60) + 'm ago';
      if (s < 86400) return Math.floor(s / 3600) + 'h ago';
      return Math.floor(s / 86400) + 'd ago';
    },

    renderCard: function(result) {
      var cardId = 'compare-card-' + result.model.replace(/[^a-zA-Z0-9_-]/g, '_');
      var card = document.getElementById(cardId);
      if (!card) return;

      var content = '';
      var isError = false;
      var inputTokens = 0;
      var outputTokens = 0;

      if (result.error) {
        content = result.error;
        isError = true;
      } else if (!result.ok) {
        content = result.data.error && result.data.error.message
          ? result.data.error.message
          : JSON.stringify(result.data.error || result.data, null, 2);
        isError = true;
      } else {
        var blocks = result.data.content || [];
        blocks.forEach(function(b) {
          if (b.type === 'text' && b.text) content += b.text;
        });
        if (!content) content = '(empty response)';
        inputTokens = (result.data.usage || {}).input_tokens || 0;
        outputTokens = (result.data.usage || {}).output_tokens || 0;
      }

      var cardContentId = 'compare-content-' + result.model.replace(/[^a-zA-Z0-9_-]/g, '_');
      card.innerHTML =
        '<div class="compare-card-header">' +
          '<div style="display:flex;align-items:center;gap:8px;flex:1;min-width:0">' +
            '<span class="compare-card-model" style="overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="' + U.escapeHtml(result.model) + '">' + U.escapeHtml(result.model) + '</span>' +
            (isError ? '<span style="color:var(--red);font-size:0.7rem">error</span>' : '<span style="color:var(--green);font-size:0.7rem">' + result.latency_ms + 'ms</span>') +
          '</div>' +
          '<div class="compare-card-actions">' +
            '<button data-copy-compare="' + cardContentId + '" title="Copy response">Copy</button>' +
          '</div>' +
        '</div>' +
        '<div class="compare-card-body' + (isError ? ' error' : '') + '" id="' + cardContentId + '">' + U.escapeHtml(content) + '</div>' +
        (isError ? '' :
          '<div class="compare-card-footer">' +
            '<span>In: ' + inputTokens + '</span>' +
            '<span>Out: ' + outputTokens + '</span>' +
            '<span>' + result.latency_ms + 'ms</span>' +
            (outputTokens > 0 ? '<span>' + (function() { var t = outputTokens / (result.latency_ms / 1000); return t >= 1 ? Math.round(t) : t.toFixed(1); })() + ' tok/s</span>' : '') +
          '</div>'
        );
    },
  };
})();
