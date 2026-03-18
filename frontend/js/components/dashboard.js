'use strict';

// ============================================================================
// SwarmLLM — Dashboard Component
// Stats, model cards, peer list, shard grid, acquisition progress
// ============================================================================

(function() {
  var S = App.state;
  var U = App.utils;

  App.dashboard = {
    _peersExpanded: false,
    _lastPeers: [],

    loadInitial: async function() {
      var statsResult, modelsResult;
      try {
        var results = await Promise.all([App.data.loadStats(), App.models.load()]);
        statsResult = results[0];
        modelsResult = results[1];
      } catch (e) {
        App.ui.showBanner('error', I18n.t('errors.server_unreachable'));
        return;
      }

      if (statsResult.stats) {
        App.dashboard.updateFull(statsResult.stats);
      } else {
        App.ui.showBanner('error', I18n.t('errors.server_unreachable'));
      }

      if (statsResult.config) {
        var cfg = statsResult.config;
        if (cfg.contribution) document.getElementById('settings-contribution').value = cfg.contribution;
        if (cfg.max_concurrent_requests) document.getElementById('settings-max-requests').value = cfg.max_concurrent_requests;
        if (cfg.max_bandwidth_mbps !== undefined) document.getElementById('settings-bandwidth').value = cfg.max_bandwidth_mbps;
        if (cfg.max_disk_mb) document.getElementById('settings-disk').value = cfg.max_disk_mb;
      }

      App.downloads.load();
      App.dashboard.loadNetworkData();
      App.networkCode.load();
    },

    updateFull: function(data) {
      if (data.node_id) document.getElementById('node-id').textContent = data.node_id;
      if (data.version) document.getElementById('version').textContent = 'v' + data.version;
      if (data.uptime_seconds !== undefined) document.getElementById('uptime').textContent = U.formatUptime(data.uptime_seconds);
      if (data.tier) {
        U.setTierBadge('tier-badge', data.tier);
        U.setTierBadge('credit-tier', data.tier);
      }

      App.dashboard.updateStats(data);

      if (data.hardware) {
        var hw = data.hardware;
        var gpuEl = document.getElementById('node-gpu');
        var gpuBadge = document.getElementById('node-gpu-badge');
        if (hw.gpu_name) {
          gpuEl.textContent = hw.gpu_name;
          if (gpuBadge) {
            if (hw.gpu_inference) {
              var backendLabel = hw.inference_backend || 'GPU';
              gpuBadge.textContent = backendLabel;
              gpuBadge.className = 'node-mode-badge node-mode-gpu';
              gpuBadge.title = 'Inference running on GPU via ' + backendLabel;
            } else {
              gpuBadge.textContent = 'CPU mode';
              gpuBadge.className = 'node-mode-badge node-mode-cpu';
              gpuBadge.title = 'GPU detected but inference running on CPU — check build flags';
            }
          }
          if (hw.gpu_vram_mb) {
            var vramUsed = hw.gpu_vram_used_mb || 0;
            var vramTotal = hw.gpu_vram_mb;
            var activeVramMb = 0;
            if (window._lastModelsData && window._lastModelsData.length) {
              window._lastModelsData.forEach(function(m) {
                if (m.status === 'loaded' && m.estimated_vram_mb) activeVramMb += m.estimated_vram_mb;
              });
            }
            var displayUsed = activeVramMb > 0 ? activeVramMb : vramUsed;
            var vramEl = document.getElementById('node-vram');
            if (activeVramMb > 0 && vramUsed > activeVramMb + 200) {
              vramEl.textContent = U.formatMB(activeVramMb) + ' active / ' + U.formatMB(vramTotal);
              vramEl.title = U.formatMB(vramUsed) + ' reserved by CUDA (freed models are cached — normal GPU behavior)';
            } else {
              vramEl.textContent = U.formatMB(displayUsed) + ' / ' + U.formatMB(vramTotal);
              vramEl.title = '';
            }
            var vramPct = vramTotal > 0 ? (displayUsed / vramTotal * 100) : 0;
            document.getElementById('vram-bar').style.width = vramPct.toFixed(1) + '%';
            document.getElementById('vram-bar').className = vramPct > 90 ? 'fill red' : (vramPct > 70 ? 'fill orange' : 'fill cyan');
          }
        } else {
          gpuEl.textContent = 'None';
          if (gpuBadge) {
            gpuBadge.textContent = 'CPU mode';
            gpuBadge.className = 'node-mode-badge node-mode-cpu';
            gpuBadge.title = 'No GPU detected — all inference runs on CPU';
          }
          document.getElementById('node-vram').textContent = '\u2014';
          document.getElementById('vram-bar').style.width = '0%';
        }
        document.getElementById('node-cpu').textContent = hw.cpu_name ? hw.cpu_name + ' (' + hw.cpu_cores + ' cores)' : 'Unknown';

        if (hw.total_ram_mb) {
          document.getElementById('ram-total').textContent = '/ ' + U.formatMB(hw.total_ram_mb);
          var ramUsed = hw.used_ram_mb || 0;
          document.getElementById('ram-used').textContent = U.formatMB(ramUsed);
          var ramPct = hw.total_ram_mb > 0 ? (ramUsed / hw.total_ram_mb * 100) : 0;
          document.getElementById('ram-bar').style.width = ramPct.toFixed(1) + '%';
          document.getElementById('ram-bar').className = ramPct > 90 ? 'fill red' : (ramPct > 70 ? 'fill orange' : 'fill green');
        }
        if (hw.total_disk_mb) {
          document.getElementById('disk-total').textContent = '/ ' + U.formatMB(hw.total_disk_mb);
          var diskUsed = hw.used_disk_mb || 0;
          document.getElementById('disk-used').textContent = U.formatMB(diskUsed);
          var diskPct = hw.total_disk_mb > 0 ? (diskUsed / hw.total_disk_mb * 100) : 0;
          document.getElementById('disk-bar').style.width = diskPct.toFixed(1) + '%';
        }
      }

      if (data.hosted_shards !== undefined) document.getElementById('hosted-shards').textContent = data.hosted_shards;
    },

    updateStats: function(data) {
      if (data.peers !== undefined) {
        document.getElementById('stat-peers').textContent = data.peers;
        var lanBadge = document.getElementById('lan-peer-badge');
        if (lanBadge) {
          if (data.lan_peers && data.lan_peers > 0) {
            lanBadge.textContent = data.lan_peers + ' LAN';
            lanBadge.style.display = 'inline-block';
          } else {
            lanBadge.style.display = 'none';
          }
        }
      }
      if (data.credits !== undefined) {
        var bal, earned, spent;
        if (typeof data.credits === 'object') {
          bal = data.credits.balance;
          earned = data.credits.lifetime_earned || 0;
          spent = data.credits.lifetime_spent || 0;
        } else {
          bal = data.credits;
          earned = 0;
          spent = 0;
        }
        document.getElementById('stat-credits').textContent = bal.toLocaleString();
        document.getElementById('credit-balance').textContent = bal.toLocaleString();
        document.getElementById('credit-earned').textContent = '+' + earned.toLocaleString();
        document.getElementById('credit-spent').textContent = '-' + spent.toLocaleString();
        var prevBal = S.creditHistory.length > 0 ? S.creditHistory[S.creditHistory.length - 1]._bal : bal;
        var delta = bal - prevBal;
        S.creditHistory.push({ _bal: bal, v: delta });
        if (S.creditHistory.length > 30) S.creditHistory.shift();
        U.renderSparkline('credit-sparkline', S.creditHistory.map(function(e) { return e.v; }));
      }
      if (data.requests_served !== undefined) document.getElementById('stat-served').textContent = data.requests_served;
      if (data.requests_made !== undefined) document.getElementById('stat-requests-made').textContent = data.requests_made;
      if (data.forwards_served !== undefined) document.getElementById('stat-forwards').textContent = data.forwards_served;
      if (data.active_requests !== undefined) document.getElementById('stat-active').textContent = data.active_requests;

      App.modeIndicator.update(data, S._cachedProviderData);

      if (typeof NeuralBg !== 'undefined') NeuralBg.updateState(data);
    },

    renderModels: function(models, cloudModels) {
      window._lastModelsData = models || [];
      var list = document.getElementById('models-list');
      var empty = document.getElementById('models-empty');
      var loading = document.getElementById('models-loading');
      if (loading) loading.remove();

      var hasCloud = cloudModels && cloudModels.length > 0;
      if ((!models || models.length === 0) && !hasCloud) {
        list.innerHTML = '';
        empty.style.display = '';
        var _sb = document.getElementById('models-stats-bar');
        if (_sb) _sb.style.display = 'none';
        return;
      }

      // Filter out ghost models
      models = models.filter(function(m) {
        if (m.local || m.hosted_shards > 0) return true;
        if (m.peers_hosting > 0) return true;
        if (m.acquisition === 'downloading') return true;
        var anyHolder = (m.shards || []).some(function(s) { return s.holders > 0; });
        return anyHolder;
      });

      if (models.length === 0 && !hasCloud) {
        list.innerHTML = '';
        empty.style.display = '';
        var _sb2 = document.getElementById('models-stats-bar');
        if (_sb2) _sb2.style.display = 'none';
        return;
      }

      empty.style.display = 'none';
      list.innerHTML = '';

      // Quick stats
      var statsBar = document.getElementById('models-stats-bar');
      if (statsBar) {
        var statLocal = models.filter(function(m) { return m.local || m.hosted_shards > 0; }).length;
        var statReady = models.filter(function(m) {
          var hc = m.hosted_shards || 0, sc = m.shard_count || (m.shards || []).length;
          return m.status === 'loaded' || m.status === 'ready' || (hc === sc && sc > 0);
        }).length;
        var statNet = models.filter(function(m) { return !m.local && !(m.hosted_shards > 0) && m.peers_hosting > 0; }).length;
        var statCloudTotal = hasCloud ? cloudModels.length : 0;
        var statProviders = 0;
        if (hasCloud) {
          var _pset = {};
          cloudModels.forEach(function(cm) { _pset[cm.provider || 'cloud'] = 1; });
          statProviders = Object.keys(_pset).length;
        }
        document.getElementById('stat-chip-ready-val').textContent = statReady;
        document.getElementById('stat-chip-network-val').textContent = statNet;
        document.getElementById('stat-chip-cloud-val').textContent = statCloudTotal;
        document.getElementById('stat-chip-providers-val').textContent = statProviders;
        statsBar.style.display = '';
        var netChip = document.getElementById('stat-chip-network');
        if (netChip) netChip.style.display = statNet > 0 ? '' : 'none';
        var cloudGroup = document.getElementById('stat-group-cloud');
        var sep = statsBar.querySelector('.models-stat-sep');
        if (cloudGroup) cloudGroup.style.display = hasCloud ? '' : 'none';
        if (sep) sep.style.display = hasCloud ? '' : 'none';
      }

      // Swarm models section
      var swarmBody;
      if (models.length > 0) {
        var swarmSection = document.createElement('details');
        swarmSection.className = 'models-section';
        swarmSection.open = true;
        var swarmReadyCount = models.filter(function(m) {
          var hc = m.hosted_shards || 0, sc = m.shard_count || (m.shards || []).length;
          return m.status === 'loaded' || m.status === 'ready' || (hc === sc && sc > 0);
        }).length;
        var swarmMeta = models.length + ' model' + (models.length !== 1 ? 's' : '') +
          (swarmReadyCount > 0 ? ' \u00b7 ' + swarmReadyCount + ' ready' : '');
        swarmSection.innerHTML = '<summary class="models-section-header">' +
          '<img src="/static/icons/swarm.svg" width="16" height="16" alt="" aria-hidden="true" class="models-section-logo">' +
          '<span class="models-section-title">Swarm Models</span>' +
          '<span class="models-section-count">' + swarmMeta + '</span>' +
          '</summary>';
        swarmBody = document.createElement('div');
        swarmBody.className = 'models-section-body';
        swarmSection.appendChild(swarmBody);
        list.appendChild(swarmSection);
      }

      models.forEach(function(m) {
        var shards = m.shards || [];
        var shardCount = m.shard_count || shards.length || 0;
        var hostedShards = m.hosted_shards || 0;
        var globalAvail = m.global_available || hostedShards;
        var isDownloading = m.acquisition === 'downloading';
        var isReady = m.status === 'loaded' || m.status === 'ready' || (globalAvail === shardCount && shardCount > 0);
        var isPartial = !isReady && hostedShards > 0 && hostedShards < shardCount;
        var safeId = (m.id || '').replace(/[^a-zA-Z0-9]/g, '_');

        var card = document.createElement('div');
        card.className = 'model-card' + (isReady ? ' ready' : (isDownloading ? ' downloading' : (isPartial ? ' partial' : '')));
        card.setAttribute('data-model-id', m.id);

        // Status pill
        var statusHtml = '';
        if (m.status === 'loaded') {
          statusHtml = '<span class="model-status-pill active">\u25CF Active</span>';
        } else if (isReady) {
          statusHtml = '<span class="model-status-pill ready">Ready</span>';
        } else if (isDownloading) {
          statusHtml = '<span class="model-status-pill downloading"><span class="spinner" style="width:10px;height:10px;border-width:1.5px;vertical-align:middle;margin-right:3px"></span>Downloading</span>';
        } else if (isPartial) {
          statusHtml = '<span class="model-status-pill partial">' + hostedShards + '/' + shardCount + ' local</span>';
        } else {
          statusHtml = '<span class="model-status-pill network">On network</span>';
        }

        // Trust level badge
        var trustBadge = '';
        if (m.trust_level === 'network_popular') {
          trustBadge = '<span class="badge-trust badge-trust-popular" title="Widely hosted across the network">Popular</span>';
        } else if (m.trust_level === 'demand_verified') {
          trustBadge = '<span class="badge-trust badge-trust-verified" title="Has received real inference requests">Verified</span>';
        } else if (m.trust_level === 'pinned') {
          trustBadge = '<span class="badge-trust badge-trust-pinned" title="Manually approved by you">Pinned</span>';
        }

        // Encrypted pipeline badge
        var encBadge = '';
        if (m.shard_count > 1 && m.local) {
          var encReady = m.has_first_shard && m.has_last_shard;
          var encActive = m.encrypted_pipeline;
          var encClass = encActive ? 'badge-encrypted active' : (encReady ? 'badge-encrypted ready' : 'badge-encrypted faded');
          var encTitle = encActive ? 'Encrypted pipeline active \u2014 click to disable' :
            (encReady ? 'Encrypted pipeline available \u2014 click to enable' :
              'Encrypted pipeline unavailable \u2014 need first + last shard');
          var missingParts = [];
          if (!m.has_first_shard) missingParts.push('first (shard 0)');
          if (!m.has_last_shard) missingParts.push('last (shard ' + (m.shard_count - 1) + ')');
          if (missingParts.length > 0) encTitle += '. Missing: ' + missingParts.join(', ');
          encBadge = '<span class="' + encClass + '" data-enc-toggle="' + U.escapeHtml(m.id) + '" data-enc-ready="' + (encReady ? '1' : '0') + '" title="' + U.escapeHtml(encTitle) + '">&#128274;</span>';
        }

        // Source label
        var sourceLabel = '';
        if (m.source === 'network' && hostedShards === 0) {
          sourceLabel = '<span class="badge badge-remote" title="Available via network peers">Remote</span>';
        }

        // Gear + info buttons
        var gearHtml = '<button class="model-gear-btn" data-am-gear="' + U.escapeHtml(m.id) + '" title="Auto-manage settings">&#9881;</button>';
        var metaBtnHtml = m.has_header ? '<button class="model-meta-btn" data-meta-toggle="' + U.escapeHtml(m.id) + '" title="GGUF Metadata">&#9432;</button>' : '';

        // Shard grid
        var shardHtml = '';
        if (shards.length > 0) {
          var lastIdx = shardCount - 1;
          var sizeClass = shardCount > 50 ? ' shard-grid-sm' : (shardCount > 20 ? ' shard-grid-md' : '');
          shardHtml = '<div class="shard-grid' + sizeClass + '" data-model-grid="' + safeId + '">';
          var localCount = 0, peerCount = 0, dlCount = 0, peerDlCount = 0, queuedCount = 0, missingCount = 0;

          shards.forEach(function(s) {
            var cls = 'missing';
            var label = '' + s.index;
            var dlPct = 0;

            if (s.local) { cls = 'local'; localCount++; }
            else if (s.holders > 0) { cls = 'peer'; peerCount++; }
            else { missingCount++; }

            if (s.download && s.download.state === 'Downloading') {
              dlPct = s.download.progress_pct || 0;
              cls = 'downloading'; dlCount++;
              label = dlPct + '%';
              if (missingCount > 0) missingCount--;
              if (peerCount > 0 && !s.local) peerCount--;
            } else if (s.download && s.download.state === 'Verifying') {
              cls = 'verifying'; dlCount++;
              label = '\u2713';
              if (missingCount > 0) missingCount--;
              if (peerCount > 0 && !s.local) peerCount--;
            } else if (s.download && (s.download.state === 'Queued' || s.download.state === 'pending')) {
              cls = 'queued'; queuedCount++;
              label = '\u2022';
              if (missingCount > 0) missingCount--;
              if (peerCount > 0 && !s.local) peerCount--;
            }

            if (s.peer_downloads && s.peer_downloads.length > 0) {
              if (cls !== 'local' && cls !== 'downloading' && cls !== 'verifying') {
                dlPct = s.peer_downloads[0].progress_pct || 0;
                cls = 'peer-downloading'; peerDlCount++;
                label = dlPct + '%';
                if (missingCount > 0) missingCount--;
                if (peerCount > 0) peerCount--;
              }
            }

            var title = 'Shard ' + s.index + (s.size_bytes ? ' (' + U.formatBytes(s.size_bytes) + ')' : '');
            if (cls === 'local') title += ' \u2014 Stored locally';
            else if (cls === 'peer') title += ' \u2014 Available from ' + s.holders + ' peer' + (s.holders !== 1 ? 's' : '');
            else if (cls === 'downloading') title += ' \u2014 Downloading (' + dlPct + '%)';
            else if (cls === 'verifying') title += ' \u2014 Verifying (BLAKE3)';
            else if (cls === 'peer-downloading') title += ' \u2014 Peer downloading (' + dlPct + '%)';
            else title += ' \u2014 Not available';

            var style = '';
            if (cls === 'downloading' || cls === 'peer-downloading') {
              style = ' style="--dl-pct:' + dlPct + '%"';
            }

            var lockIcon = s.locked ? '<span class="shard-lock-icon" title="Locked (pinned)">\uD83D\uDD12</span>' : '';

            var endpointClass = '';
            if (shardCount > 1 && (s.index === 0 || s.index === lastIdx)) {
              if (m.encrypted_pipeline && s.local) {
                endpointClass = ' shard-pinned';
              } else {
                endpointClass = ' shard-endpoint';
              }
            }

            shardHtml += '<div class="shard-cell ' + cls + (s.locked ? ' locked' : '') + endpointClass + '"' + style +
              ' data-shard="' + safeId + '-' + s.index + '"' +
              ' data-shard-model="' + U.escapeHtml(m.id) + '"' +
              ' data-shard-index="' + s.index + '"' +
              ' data-shard-locked="' + (s.locked ? '1' : '0') + '"' +
              ' title="' + U.escapeHtml(title) + '">' + label + lockIcon + '</div>';
          });
          shardHtml += '</div>';

          // Shard summary counts
          var summaryParts = [];
          if (localCount > 0) summaryParts.push('<span class="shard-sum-item shard-sum-local"><span class="shard-sum-dot"></span>' + localCount + ' local</span>');
          if (peerCount > 0) summaryParts.push('<span class="shard-sum-item shard-sum-peer"><span class="shard-sum-dot"></span>' + peerCount + ' peer' + (peerCount !== 1 ? 's' : '') + '</span>');
          if (dlCount > 0) summaryParts.push('<span class="shard-sum-item shard-sum-dl"><span class="shard-sum-dot"></span>' + dlCount + ' downloading</span>');
          if (peerDlCount > 0) summaryParts.push('<span class="shard-sum-item shard-sum-peer-dl"><span class="shard-sum-dot"></span>' + peerDlCount + ' peer DL</span>');
          if (queuedCount > 0) summaryParts.push('<span class="shard-sum-item shard-sum-queued"><span class="shard-sum-dot"></span>' + queuedCount + ' queued</span>');
          if (missingCount > 0) summaryParts.push('<span class="shard-sum-item shard-sum-missing"><span class="shard-sum-dot"></span>' + missingCount + ' missing</span>');
          if (summaryParts.length > 0) {
            shardHtml += '<div class="shard-summary" data-model-summary="' + safeId + '">' + summaryParts.join('') + '</div>';
          }
        }

        // Download progress bar
        var progressHtml = '';
        if (isDownloading && m.acquisition_progress) {
          var ap = m.acquisition_progress;
          var dlBytes = ap.downloaded_bytes || 0;
          var totalBytes = ap.total_bytes || 0;
          if (dlBytes > totalBytes && totalBytes > 0) dlBytes = totalBytes;
          var pct = totalBytes > 0 ? Math.min(100, Math.round((dlBytes / totalBytes) * 100)) : 0;
          var speed = ap.speed_bytes_per_sec || 0;
          var etaStr = '';
          if (speed > 0 && totalBytes > dlBytes) {
            etaStr = U.formatEta((totalBytes - dlBytes) / speed);
          }
          var dlShards2 = shards.filter(function(s) { return s.download || s.local; });
          var segmentCount = Math.max(dlShards2.length, shardCount);
          var segmentsHtml = '';
          if (segmentCount > 0) {
            var segW = (100 / segmentCount);
            for (var si = 0; si < segmentCount; si++) {
              var sh = shards.find(function(s) { return s.index === si; });
              var segPct = 0;
              if (sh && sh.local) segPct = 100;
              else if (sh && sh.download) segPct = sh.download.progress_pct || 0;
              segmentsHtml += '<div class="dl-seg" style="width:' + segW.toFixed(2) + '%;"><div class="dl-seg-fill" style="width:' + segPct + '%"></div></div>';
            }
          }
          var shardLabel = ap.downloaded_shards !== undefined ? ('Shard ' + ap.downloaded_shards + '/' + shardCount) : 'Downloading';
          var rightText = U.formatBytes(dlBytes) + ' / ' + U.formatBytes(totalBytes) + ' (' + pct + '%)';
          if (speed > 0) rightText += ' &middot; ' + U.formatSpeed(speed);
          if (etaStr) rightText += ' &middot; ETA ' + etaStr;
          progressHtml = '<div class="dl-progress" data-model-progress="' + safeId + '" data-last-pct="' + pct + '">' +
            '<div class="flex-between" style="font-size:0.75rem;margin-bottom:3px">' +
            '<span class="text-muted">' + shardLabel + '</span>' +
            '<span class="mono dl-progress-text">' + rightText + '</span>' +
            '</div>' +
            '<div class="dl-bar">' + (segmentsHtml || '<div class="dl-fill" style="width:' + pct + '%"></div>') + '</div>' +
            '</div>';
        }

        // Per-shard download bars
        var perShardDlHtml = '';
        if (isDownloading && shards.length > 0 && shardCount <= 20) {
          var dlShardBars = shards.filter(function(s) {
            return s.download && s.download.state === 'Downloading';
          });
          if (dlShardBars.length > 1) {
            perShardDlHtml = '<div class="per-shard-dl">';
            dlShardBars.forEach(function(s) {
              var pct2 = s.download.progress_pct || 0;
              var bytes = s.download.downloaded_bytes || 0;
              var total = s.download.total_bytes || s.size_bytes || 0;
              perShardDlHtml += '<div class="per-shard-dl-row">' +
                '<span class="per-shard-dl-label">Shard ' + s.index + '</span>' +
                '<div class="per-shard-dl-bar"><div class="per-shard-dl-fill" style="width:' + pct2 + '%"></div></div>' +
                '<span class="per-shard-dl-pct">' + U.formatBytes(bytes) + '/' + U.formatBytes(total) + ' (' + pct2 + '%)</span>' +
                '</div>';
            });
            perShardDlHtml += '</div>';
          }
        }

        // Footer meta info
        var footerMeta = [];
        footerMeta.push(U.formatBytes(m.total_size_bytes || 0));
        if (shardCount > 0) footerMeta.push(shardCount + (shardCount === 1 ? ' shard' : ' shards'));
        if (m.estimated_vram_mb) footerMeta.push('~' + U.formatMB(m.estimated_vram_mb) + ' VRAM');
        if (m.peers_hosting > 0) footerMeta.push(m.peers_hosting + ' peer' + (m.peers_hosting !== 1 ? 's' : ''));
        else if (hostedShards > 0) footerMeta.push('<span style="color:var(--orange)">Local only</span>');

        // Missing files warning
        var fileIndicators = '';
        if (hostedShards > 0 || isDownloading) {
          var hasManifest = m.has_manifest !== false;
          var hasHeader = m.has_header !== false;
          if (!hasManifest || !hasHeader) {
            var missingFiles = [];
            if (!hasManifest) missingFiles.push('manifest');
            if (!hasHeader) missingFiles.push('header');
            fileIndicators = '<span style="color:var(--orange);font-size:0.7rem" title="Missing: ' + missingFiles.join(', ') + '">&#9888; Missing ' + missingFiles.join(' + ') + '</span>';
          }
        }

        // Action buttons
        var actionHtml = '';
        if (m.status === 'loaded') {
          actionHtml = '<button class="btn btn-sm btn-outline" data-unload-model="' + U.escapeHtml(m.id) + '">Unload</button>';
        } else if (isReady) {
          actionHtml = '<button class="btn btn-sm btn-primary" data-select-model="' + U.escapeHtml(m.id) + '">Use</button>';
        } else if (isDownloading) {
          actionHtml = '<button class="shard-cancel-btn" data-cancel-download="' + U.escapeHtml(m.id) + '" title="Cancel download">&times; Cancel</button>';
        } else if (m.source === 'network' || m.status === 'available' || m.status === 'partial') {
          actionHtml = '<button class="btn btn-sm" data-request-model="' + U.escapeHtml(m.id) + '">Download</button>';
        }

        var removeHtml = '';
        if (hostedShards > 0 && !isDownloading) {
          removeHtml = '<button class="model-remove-btn" data-remove-model="' + U.escapeHtml(m.id) + '">Remove</button>';
        }

        var name = U.formatModelDisplayName(m.name || m.id);
        var creatorIconHtml = providerIconHtml(modelIconKey(m.id), 14);

        // Card HTML
        card.innerHTML =
          '<div class="model-card-title">' +
            '<div class="model-card-name-row">' +
              creatorIconHtml +
              '<span class="model-name" title="' + U.escapeHtml(m.id) + '">' + U.escapeHtml(name) + '</span>' +
              encBadge + sourceLabel + trustBadge +
            '</div>' +
            '<div class="model-card-controls">' +
              statusHtml + metaBtnHtml + gearHtml +
            '</div>' +
          '</div>' +
          '<div class="model-card-shards">' +
            shardHtml + progressHtml + perShardDlHtml +
          '</div>' +
          '<div class="model-card-footer">' +
            '<div class="model-card-meta">' +
              footerMeta.map(function(p) { return '<span>' + p + '</span>'; }).join('') +
              (fileIndicators ? '<span>' + fileIndicators + '</span>' : '') +
            '</div>' +
            '<div class="model-card-actions">' + actionHtml + removeHtml + '</div>' +
          '</div>' +
          '<div class="gguf-metadata-panel hidden" data-meta-panel="' + U.escapeHtml(m.id) + '"></div>';

        if (swarmBody) swarmBody.appendChild(card);
      });

      // Cloud provider models
      if (hasCloud) {
        var byProvider = {};
        cloudModels.forEach(function(cm) {
          var p = cm.provider || 'cloud';
          if (!byProvider[p]) byProvider[p] = [];
          byProvider[p].push(cm);
        });

        function getCtxLen(cm) {
          if (!cm.meta) return 0;
          return cm.meta.context_length || cm.meta.context_window || cm.meta.max_model_len || 0;
        }

        var _nonChatPattern = /dall-e|tts|whisper|embed|moderation|davinci-\d|babbage-\d|text-embedding|audio/i;

        function sortCloudModels(models, sortBy) {
          var sorted = models.slice();
          if (sortBy === 'ctx-desc') {
            sorted.sort(function(a, b) { return getCtxLen(b) - getCtxLen(a); });
          } else if (sortBy === 'ctx-asc') {
            sorted.sort(function(a, b) { return getCtxLen(a) - getCtxLen(b); });
          } else if (sortBy === 'avail') {
            sorted.sort(function(a, b) {
              var sa = S.modelStatus[a.id], sb = S.modelStatus[b.id];
              var rank = { up: 0, rate_limited: 1, timeout: 3, unavailable: 4, not_found: 5, error: 4 };
              var ra = sa ? (rank[sa.status] !== undefined ? rank[sa.status] : 2) : 2;
              var rb = sb ? (rank[sb.status] !== undefined ? rank[sb.status] : 2) : 2;
              if (ra !== rb) return ra - rb;
              var la = sa ? sa.latency_ms : 99999, lb = sb ? sb.latency_ms : 99999;
              return la - lb;
            });
          } else if (sortBy === 'popular') {
            sorted.sort(function(a, b) {
              var aNon = _nonChatPattern.test(a.id) ? 1 : 0;
              var bNon = _nonChatPattern.test(b.id) ? 1 : 0;
              if (aNon !== bNon) return aNon - bNon;
              var ca = (a.meta && a.meta.created) || 0;
              var cb = (b.meta && b.meta.created) || 0;
              if (ca !== cb) return cb - ca;
              var na = (a.name || a.id).toLowerCase(), nb = (b.name || b.id).toLowerCase();
              return na < nb ? -1 : na > nb ? 1 : 0;
            });
          } else {
            sorted.sort(function(a, b) {
              var na = (a.name || a.id).toLowerCase(), nb = (b.name || b.id).toLowerCase();
              return na < nb ? -1 : na > nb ? 1 : 0;
            });
          }
          return sorted;
        }

        function renderCloudRow(cm) {
          var ctx = getCtxLen(cm);
          var ctxStr = ctx > 0 ? (ctx >= 1000 ? Math.round(ctx / 1000) + 'K' : ctx.toString()) : '';
          var pingHtml = App.providerHealth.modelBadgeHtml(cm.id);
          return '<div class="cloud-model-row" data-select-cloud="' + U.escapeHtml(cm.id) + '" title="' + U.escapeHtml(cm.id) + '">' +
            '<span class="cloud-model-row-name">' + U.escapeHtml(cm.name || cm.id) + '</span>' +
            (ctxStr ? '<span class="cloud-model-row-ctx">' + ctxStr + '</span>' : '<span class="cloud-model-row-ctx"></span>') +
            '<span class="cloud-model-row-ping">' + pingHtml + '</span>' +
            '</div>';
        }

        function renderRowsInto(container, models) {
          container.innerHTML = models.length > 0
            ? models.map(renderCloudRow).join('')
            : '<div class="cloud-model-empty">No models match</div>';
        }

        var providerCount = Object.keys(byProvider).length;
        var cloudSection = document.createElement('details');
        cloudSection.className = 'models-section';
        cloudSection.open = true;
        var cloudMeta = providerCount + ' provider' + (providerCount !== 1 ? 's' : '') +
          ' \u00b7 ' + cloudModels.length + ' models';
        cloudSection.innerHTML = '<summary class="models-section-header">' +
          '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" aria-hidden="true" class="models-section-logo" style="flex-shrink:0"><path d="M18 10h-1.26A8 8 0 1 0 9 20h9a5 5 0 0 0 0-10z" fill="var(--accent)"/></svg>' +
          '<span class="models-section-title">Cloud Providers</span>' +
          '<span class="models-section-count">' + cloudMeta + '</span>' +
          '</summary>';
        var cloudBody = document.createElement('div');
        cloudBody.className = 'models-section-body';
        cloudSection.appendChild(cloudBody);
        list.appendChild(cloudSection);

        Object.keys(byProvider).forEach(function(p) {
          var pLabel = PROVIDER_NAMES[p] || p;
          var pModels = byProvider[p];
          var sorted = sortCloudModels(pModels, 'popular');
          var filterId = 'cloud-filter-' + p;
          var sortId = 'cloud-sort-' + p;
          var listId = 'cloud-list-wrap-' + p;

          var card = document.createElement('div');
          card.className = 'model-card cloud-model';
          card.setAttribute('data-provider', p);

          var cardIconHtml = providerIconHtml(p, 18);
          card.innerHTML =
            '<div class="cloud-card-header">' +
              '<span class="cloud-provider-name">' + (cardIconHtml ? cardIconHtml + ' ' : '') + U.escapeHtml(pLabel) + '</span>' +
              '<span>' +
                '<span class="badge badge-cloud">' + pModels.length + ' model' + (pModels.length !== 1 ? 's' : '') + '</span>' +
                '<span class="cloud-status-ok">\u25cf Connected</span>' +
              '</span>' +
            '</div>' +
            '<div class="cloud-card-controls">' +
              '<input type="text" class="cloud-model-filter" id="' + filterId + '" placeholder="Search models\u2026" autocomplete="off">' +
              '<select class="cloud-model-sort" id="' + sortId + '">' +
                '<option value="popular">Newest</option>' +
                '<option value="az">A\u2013Z</option>' +
                '<option value="ctx-desc">Context \u2193</option>' +
                '<option value="ctx-asc">Context \u2191</option>' +
                '<option value="avail">Ping \u2193</option>' +
              '</select>' +
            '</div>' +
            '<div class="cloud-model-list" id="' + listId + '"></div>' +
            '<div class="cloud-card-note">Requests routed to ' + U.escapeHtml(pLabel) + ' API \u2014 not shared on the swarm network</div>';

          cloudBody.appendChild(card);

          var listContainer = document.getElementById(listId);
          if (listContainer) renderRowsInto(listContainer, sorted);

          var visibleIds = sorted.slice(0, 20).map(function(cm) { return cm.id; });
          setTimeout(function() { App.providerHealth.probe(visibleIds); }, 500);

          var filterEl = document.getElementById(filterId);
          var sortEl = document.getElementById(sortId);
          var refreshRows = function() {
            var query = filterEl ? filterEl.value.toLowerCase().trim() : '';
            var sortBy = sortEl ? sortEl.value : 'popular';
            var filtered = query ? pModels.filter(function(cm) {
              var text = ((cm.name || '') + ' ' + cm.id + ' ' + (cm.meta && cm.meta.owned_by ? cm.meta.owned_by : '')).toLowerCase();
              return text.indexOf(query) !== -1;
            }) : pModels;
            var s = sortCloudModels(filtered, sortBy);
            if (listContainer) renderRowsInto(listContainer, s);
            App.providerHealth.probe(s.slice(0, 20).map(function(cm) { return cm.id; }));
          };
          if (filterEl) {
            filterEl.addEventListener('input', refreshRows);
            filterEl.addEventListener('paste', function() { setTimeout(refreshRows, 0); });
          }
          if (sortEl) sortEl.addEventListener('change', function() {
            refreshRows();
            if (sortEl.value === 'avail') App.providerHealth.probe(pModels.map(function(cm) { return cm.id; }).slice(0, 40));
          });
        });
        if (Object.keys(S.modelStatus).length > 0) App.providerHealth.updateModelBadges();
      }
    },

    updateShardsLive: function(acquisitions, shardRegistry, peerDownloads) {
      if (!acquisitions && !shardRegistry && !peerDownloads) return;

      if (acquisitions) {
        acquisitions.forEach(function(acq) {
          var modelId = acq.model_id;
          if (!modelId) return;
          var safeId = modelId.replace(/[^a-zA-Z0-9]/g, '_');

          var shardDetails = acq.shard_details || [];
          var localCount = 0, peerCount = 0, dlCount = 0, peerDlCount = 0, queuedCount = 0, missingCount = 0;
          shardDetails.forEach(function(sd) {
            var cellId = safeId + '-' + sd.index;
            var cell = document.querySelector('[data-shard="' + cellId + '"]');
            if (!cell) return;

            var oldClass = cell.className.replace(/shard-cell\s*/, '').trim().split(/\s+/)[0] || 'missing';
            var newClass = 'missing';
            var label = '' + sd.index;
            var dlPct = sd.progress_pct || 0;

            if (sd.state === 'complete') { newClass = 'local'; localCount++; }
            else if (sd.state === 'verifying') {
              newClass = 'verifying'; dlCount++;
              label = '\u2713';
            } else if (sd.state === 'downloading') {
              newClass = 'downloading'; dlCount++;
              label = dlPct + '%';
            } else if (sd.state === 'pending') {
              newClass = 'queued'; queuedCount++;
              label = '\u2022';
            } else if (sd.state === 'failed') { newClass = 'missing'; missingCount++; }
            else { missingCount++; }

            if (oldClass !== newClass || cell.textContent !== label) {
              cell.className = 'shard-cell ' + newClass;
              cell.textContent = label;

              if (newClass === 'downloading' || newClass === 'peer-downloading') {
                cell.style.setProperty('--dl-pct', dlPct + '%');
              } else {
                cell.style.removeProperty('--dl-pct');
              }

              var title = 'Shard ' + sd.index;
              if (newClass === 'local') title += ' \u2014 Verified, stored locally';
              else if (newClass === 'verifying') title += ' \u2014 Downloaded, verifying (BLAKE3)...';
              else if (newClass === 'downloading') title += ' \u2014 Downloading (' + dlPct + '%)';
              else if (newClass === 'queued') title += ' \u2014 Queued for download';
              else if (sd.state === 'failed') title += ' \u2014 Failed';
              else title += ' \u2014 Not available';
              cell.setAttribute('title', title);
            }
          });

          // Update progress bar
          var progressEl = document.querySelector('[data-model-progress="' + safeId + '"]');
          if (progressEl && acq.total_bytes > 0) {
            var dlBytes = Math.min(acq.downloaded_bytes || 0, acq.total_bytes);
            var pct = Math.min(100, Math.round((dlBytes / acq.total_bytes) * 100));
            var lastPct = parseInt(progressEl.getAttribute('data-last-pct') || '0', 10);
            if (pct >= lastPct) {
              progressEl.setAttribute('data-last-pct', '' + pct);
              var speed = acq.speed_bytes_per_sec || 0;
              var shardLabel = acq.downloaded_shards !== undefined ? ('Shard ' + acq.downloaded_shards + '/' + (acq.total_shards || shardDetails.length)) : 'Downloading';
              var etaStr = '';
              if (speed > 0 && acq.total_bytes > dlBytes) {
                etaStr = U.formatEta((acq.total_bytes - dlBytes) / speed);
              }
              var textEl = progressEl.querySelector('.dl-progress-text');
              if (textEl) {
                var txt = U.formatBytes(dlBytes) + ' / ' + U.formatBytes(acq.total_bytes) + ' (' + pct + '%)';
                if (speed > 0) txt += ' \u00b7 ' + U.formatSpeed(speed);
                if (etaStr) txt += ' \u00b7 ETA ' + etaStr;
                textEl.textContent = txt;
              }
              var labelEl = progressEl.querySelector('.text-muted');
              if (labelEl) labelEl.textContent = shardLabel;
              var segs = progressEl.querySelectorAll('.dl-seg');
              if (segs.length > 0) {
                shardDetails.forEach(function(sd) {
                  if (segs[sd.index]) {
                    var segFill = segs[sd.index].querySelector('.dl-seg-fill');
                    var segPct = sd.state === 'complete' ? 100 : (sd.progress_pct || 0);
                    if (segFill) segFill.style.width = segPct + '%';
                  }
                });
              } else {
                var fillEl = progressEl.querySelector('.dl-fill');
                if (fillEl) fillEl.style.width = pct + '%';
              }
            }
          } else if (!progressEl && acq.total_bytes > 0 && acq.downloaded_bytes > 0) {
            var card = document.querySelector('[data-model-id="' + U.cssSafeAttr(modelId) + '"]');
            if (card && !card.querySelector('.dl-progress')) {
              var dlBytes2 = Math.min(acq.downloaded_bytes, acq.total_bytes);
              var pct2 = Math.min(100, Math.round((dlBytes2 / acq.total_bytes) * 100));
              var speed2 = acq.speed_bytes_per_sec || 0;
              var shardLabel2 = acq.downloaded_shards !== undefined ? ('Shard ' + acq.downloaded_shards + '/' + (acq.total_shards || '?')) : 'Downloading';
              var progDiv = document.createElement('div');
              progDiv.className = 'dl-progress';
              progDiv.setAttribute('data-model-progress', safeId);
              progDiv.setAttribute('data-last-pct', '' + pct2);
              progDiv.innerHTML =
                '<div class="flex-between" style="font-size:0.75rem;margin-bottom:3px">' +
                '<span class="text-muted">' + shardLabel2 + '</span>' +
                '<span class="mono dl-progress-text">' + U.formatBytes(dlBytes2) + ' / ' + U.formatBytes(acq.total_bytes) + ' (' + pct2 + '%)' +
                (speed2 > 0 ? ' \u2014 ' + U.formatSpeed(speed2) : '') + '</span></div>' +
                '<div class="dl-bar"><div class="dl-fill" style="width:' + pct2 + '%"></div></div>';
              card.appendChild(progDiv);
              if (!card.classList.contains('downloading')) {
                card.classList.remove('partial');
                card.classList.add('downloading');
              }
            }
          }

          // Update shard summary
          var summaryEl = document.querySelector('[data-model-summary="' + safeId + '"]');
          if (summaryEl && shardDetails.length > 0) {
            var summParts = [];
            if (localCount > 0) summParts.push('<span class="shard-sum-item shard-sum-local"><span class="shard-sum-dot"></span>' + localCount + ' local</span>');
            if (peerCount > 0) summParts.push('<span class="shard-sum-item shard-sum-peer"><span class="shard-sum-dot"></span>' + peerCount + ' peer' + (peerCount !== 1 ? 's' : '') + '</span>');
            if (dlCount > 0) summParts.push('<span class="shard-sum-item shard-sum-dl"><span class="shard-sum-dot"></span>' + dlCount + ' downloading</span>');
            if (peerDlCount > 0) summParts.push('<span class="shard-sum-item shard-sum-peer-dl"><span class="shard-sum-dot"></span>' + peerDlCount + ' peer DL</span>');
            if (queuedCount > 0) summParts.push('<span class="shard-sum-item shard-sum-queued"><span class="shard-sum-dot"></span>' + queuedCount + ' queued</span>');
            if (missingCount > 0) summParts.push('<span class="shard-sum-item shard-sum-missing"><span class="shard-sum-dot"></span>' + missingCount + ' missing</span>');
            summaryEl.innerHTML = summParts.join('');
          }
        });
      }

      // Update shard cells from shard registry changes
      if (shardRegistry) {
        Object.keys(shardRegistry).forEach(function(modelId) {
          var safeId = modelId.replace(/[^a-zA-Z0-9]/g, '_');
          var shards = shardRegistry[modelId] || [];
          shards.forEach(function(s) {
            var cellId = safeId + '-' + s.index;
            var cell = document.querySelector('[data-shard="' + cellId + '"]');
            if (!cell) return;

            var current = cell.className;
            if (current.indexOf('downloading') >= 0 || current.indexOf('local') >= 0) return;

            if (s.local) {
              cell.className = 'shard-cell local';
              cell.textContent = '' + s.index;
              cell.setAttribute('title', 'Shard ' + s.index + ' \u2014 Stored locally');
            } else if (s.holders > 0 && current.indexOf('peer') < 0) {
              cell.className = 'shard-cell peer';
              cell.setAttribute('title', 'Shard ' + s.index + ' \u2014 Available from ' + s.holders + ' peer(s)');
            }
          });
        });
      }

      // Update shard cells with peer download progress
      if (peerDownloads && peerDownloads.length > 0) {
        peerDownloads.forEach(function(pd) {
          var safeId = pd.model_id.replace(/[^a-zA-Z0-9]/g, '_');
          var cellId = safeId + '-' + pd.shard_index;
          var cell = document.querySelector('[data-shard="' + cellId + '"]');
          if (!cell) return;

          var current = cell.className;
          if (current.indexOf('local') >= 0 || current.indexOf(' downloading') >= 0) return;

          var pdPct = pd.progress_pct || 0;
          cell.className = 'shard-cell peer-downloading';
          cell.style.setProperty('--dl-pct', pdPct + '%');
          cell.textContent = pdPct + '%';
          cell.setAttribute('title', 'Shard ' + pd.shard_index + ' \u2014 Peer ' + pd.node_id.substring(0, 8) + ' downloading (' + pdPct + '%)');
        });
      }
    },

    renderPeerItem: function(p) {
      var tmpl = document.getElementById('tmpl-peer-row');
      var node = tmpl.content.cloneNode(true);
      var div = node.querySelector('.peer-row-item');

      var dot = node.querySelector('.status-dot');
      dot.classList.add(p.healthy ? 'online' : 'degraded');

      var lanBadge = node.querySelector('.peer-lan-badge');
      if (p.is_lan_peer) {
        lanBadge.removeAttribute('hidden');
        lanBadge.textContent = 'LAN';
      }

      var label = node.querySelector('.peer-label');
      if (p.nickname) {
        label.textContent = p.nickname;
        var sub = document.createElement('span');
        sub.className = 'text-muted mono';
        sub.style.fontSize = '0.65rem';
        sub.textContent = ' (' + (p.node_id || '').substring(0, 8) + ')';
        label.appendChild(sub);
      } else {
        label.className = 'peer-label mono';
        label.textContent = (p.node_id || 'unknown').substring(0, 16);
      }

      var gpu = node.querySelector('.peer-gpu');
      if (p.gpu) {
        gpu.textContent = p.gpu;
      } else {
        gpu.remove();
      }

      return div;
    },

    renderPeers: function(peers) {
      var PEER_LIMIT = 5;
      var list = document.getElementById('peers-list');
      var summary = document.getElementById('peers-summary');
      var overflow = document.getElementById('peers-overflow');
      var pLoading = document.getElementById('peers-loading');
      if (pLoading) pLoading.remove();
      if (!list) return;

      App.dashboard._lastPeers = peers || [];

      if (peers && peers.length > 0) {
        var lanCount = peers.filter(function(p) { return p.is_lan_peer; }).length;
        var healthyCount = peers.filter(function(p) { return p.healthy; }).length;
        if (summary) {
          summary.textContent = peers.length + ' peer' + (peers.length !== 1 ? 's' : '') +
            (lanCount > 0 ? ' \u00B7 ' + lanCount + ' LAN' : '') +
            ' \u00B7 ' + healthyCount + ' healthy';
        }

        list.innerHTML = '';
        var showAll = App.dashboard._peersExpanded;
        var visible = showAll ? peers : peers.slice(0, PEER_LIMIT);
        visible.forEach(function(p) {
          list.appendChild(App.dashboard.renderPeerItem(p));
        });

        if (overflow) {
          if (peers.length > PEER_LIMIT && !showAll) {
            overflow.style.display = '';
            var btn = document.getElementById('btn-show-all-peers');
            if (btn) btn.textContent = 'Show all ' + peers.length + ' peers';
          } else {
            overflow.style.display = 'none';
          }
        }
      } else {
        if (summary) summary.textContent = '';
        list.innerHTML = '<div class="text-muted" style="font-size:0.85rem">' + I18n.t('network.no_peers_yet') + '</div>';
        if (overflow) overflow.style.display = 'none';
      }
    },

    loadNetworkData: async function() {
      try {
        var resp = await App.authFetch('/api/admin/peers');
        if (!resp.ok) throw new Error('fetch failed');
        var peers = await resp.json();
        App.dashboard.renderPeers(peers);
      } catch (e) {
        var list = document.getElementById('peers-list');
        var pLoading2 = document.getElementById('peers-loading');
        if (pLoading2) pLoading2.remove();
        if (list) list.innerHTML = '<div class="text-muted" style="font-size:0.85rem">' + I18n.t('network.no_peers_yet') + '</div>';
      }
    },

    updateAcquisitionProgress: function(acquisitions) {
      if (!acquisitions || acquisitions.length === 0) return;
      acquisitions.forEach(function(status) {
        var modelId = status.model_id;
        if (!modelId) return;
        if (!S.activeAcquisitions[modelId]) {
          if (status.state === 'complete') return;
          S.activeAcquisitions[modelId] = { started: Date.now() };
        }
        App.dashboard.renderAcquisitionPanel(modelId, status);

        if (status.state === 'downloading' && status.source === 'huggingface') {
          var acqInfo = S.activeAcquisitions[modelId];
          var speed = status.speed_bytes_per_sec || 0;
          if (speed > 0 && speed < 102400) {
            if (!acqInfo._slowSince) acqInfo._slowSince = Date.now();
            else if (Date.now() - acqInfo._slowSince > 30000 && !acqInfo._throttleWarned) {
              acqInfo._throttleWarned = true;
              App.notifications.showToast('Download is slow (' + U.formatSpeed(speed) + ') \u2014 this can happen with popular models. It will keep going.', 'warning', 10000);
            }
          } else {
            acqInfo._slowSince = null;
          }
        }

        if (status.state === 'complete') {
          App.notifications.showToast('Download complete: ' + (status.model_name || modelId), 'success');
          setTimeout(function() { delete S.activeAcquisitions[modelId]; App.dashboard.loadInitial(); }, 3000);
        } else if (status.state === 'failed') {
          var reason = (typeof status.state === 'object' && status.state.failed) ? status.state.failed.reason : '';
          App.notifications.showToast('Download failed: ' + (status.model_name || modelId) + (reason ? ' \u2014 ' + reason : ''), 'error', 8000);
          setTimeout(function() { delete S.activeAcquisitions[modelId]; }, 10000);
        }
      });
    },

    renderAcquisitionPanel: function(modelId, status) {
      if (!status) return;
      if (!S.activeAcquisitions[modelId]) return;
      var safeId = modelId.replace(/[^a-zA-Z0-9]/g, '_');
      var card = document.querySelector('[data-model-id="' + U.cssSafeAttr(modelId) + '"]');
      if (!card) {
        App.models.load();
        App.dashboard.loadInitial();
        return;
      }

      var stateName = typeof status.state === 'string' ? status.state : 'unknown';

      if (stateName === 'complete') {
        if (!card.classList.contains('ready')) {
          setTimeout(function() { App.dashboard.loadInitial(); }, 1500);
        }
        return;
      }

      if (!card.classList.contains('downloading')) {
        card.classList.remove('partial');
        card.classList.add('downloading');
      }

      var totalBytes = status.total_bytes || 0;
      var dlBytes = status.downloaded_bytes || 0;
      var pct = totalBytes > 0 ? Math.round((dlBytes / totalBytes) * 100) : 0;
      var speed = status.speed_bytes_per_sec || 0;

      var progressEl = card.querySelector('.dl-progress');
      if (!progressEl) {
        progressEl = document.createElement('div');
        progressEl.className = 'dl-progress';
        progressEl.setAttribute('data-model-progress', safeId);
        card.appendChild(progressEl);
      }

      var speedStr = speed > 0 ? ' - ' + U.formatSpeed(speed) : '';
      progressEl.innerHTML =
        '<div class="flex-between" style="font-size:0.75rem;margin-bottom:3px">' +
        '<span class="text-muted">Downloading model data</span>' +
        '<span style="display:flex;align-items:center;gap:8px">' +
          '<span class="mono dl-progress-text">' + U.formatBytes(dlBytes) + ' / ' + U.formatBytes(totalBytes) + ' (' + pct + '%)' + speedStr + '</span>' +
          '<button class="btn btn-sm" style="padding:1px 6px;font-size:0.7rem;line-height:1.2" data-cancel-download="' + U.escapeHtml(modelId) + '" title="Cancel download">&times; Cancel</button>' +
        '</span>' +
        '</div>' +
        '<div class="dl-bar"><div class="dl-fill" style="width:' + pct + '%"></div></div>';

      var oldPanel = document.getElementById('acq-panel-' + safeId);
      if (oldPanel) oldPanel.remove();
    }
  };
})();
