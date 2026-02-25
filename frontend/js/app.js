'use strict';

var ws = null;
var creditHistory = [];

// --- Initial data load ---

async function loadInitialData() {
  try {
    var resp = await fetch('/api/admin/stats');
    var data = await resp.json();
    updateDashboard(data);
  } catch (e) {
    showBanner('error', 'Failed to connect to SwarmLLM daemon');
  }

  try {
    var resp = await fetch('/api/admin/config');
    var cfg = await resp.json();
    if (cfg.contribution) {
      document.getElementById('settings-contribution').value = cfg.contribution;
    }
    if (cfg.max_concurrent_requests) {
      document.getElementById('settings-max-requests').value = cfg.max_concurrent_requests;
    }
    if (cfg.max_bandwidth_mbps !== undefined) {
      document.getElementById('settings-bandwidth').value = cfg.max_bandwidth_mbps;
    }
    if (cfg.max_disk_mb) {
      document.getElementById('settings-disk').value = cfg.max_disk_mb;
    }
  } catch (e) {
    console.error('Failed to load config:', e);
  }

  try {
    var resp = await fetch('/api/admin/models');
    var models = await resp.json();
    renderModelsTable(models);
  } catch (e) {
    console.error('Failed to load models:', e);
  }

  // Load governance data
  loadGovernanceData();
  loadNetworkData();
}

// --- Status banner ---

function showBanner(type, message) {
  var banner = document.getElementById('status-banner');
  if (!banner) return;
  banner.innerHTML = '<div class="alert alert-' + type + '">' + message + '</div>';
}

// --- WebSocket for real-time updates ---

function connectWebSocket() {
  var protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  ws = new WebSocket(protocol + '//' + window.location.host + '/api/admin/ws');

  ws.onmessage = function (event) {
    try {
      var msg = JSON.parse(event.data);
      handleWsMessage(msg);
    } catch (e) {}
  };

  ws.onclose = function () {
    setTimeout(connectWebSocket, 3000);
  };

  ws.onerror = function () {
    ws.close();
  };
}

function handleWsMessage(msg) {
  switch (msg.type) {
    case 'stats_update':
      updateStats(msg.data);
      if (msg.data.acquisitions) {
        updateAcquisitionProgress(msg.data.acquisitions);
      }
      break;
  }
}

// --- Dashboard update ---

function updateDashboard(data) {
  if (data.node_id) {
    document.getElementById('node-id').textContent = data.node_id;
  }
  if (data.version) {
    document.getElementById('version').textContent = 'v' + data.version;
  }
  if (data.uptime_seconds !== undefined) {
    document.getElementById('uptime').textContent = formatUptime(data.uptime_seconds);
  }
  if (data.tier) {
    setTierBadge('tier-badge', data.tier);
    setTierBadge('credit-tier', data.tier);
  }

  updateStats(data);

  if (data.hardware) {
    var hw = data.hardware;

    // GPU display
    if (hw.gpu_name) {
      document.getElementById('node-gpu').textContent = hw.gpu_name;
      if (hw.gpu_vram_mb) {
        document.getElementById('node-vram').textContent = formatMB(hw.gpu_vram_mb) + ' VRAM';
      }
    } else {
      document.getElementById('node-gpu').textContent = 'CPU only';
      document.getElementById('node-vram').textContent = '';
    }

    // CPU display
    document.getElementById('node-cpu').textContent = hw.cpu_name
      ? hw.cpu_name + ' (' + hw.cpu_cores + ' cores)'
      : 'Unknown';

    if (hw.total_ram_mb) {
      document.getElementById('ram-total').textContent = '/ ' + formatMB(hw.total_ram_mb);
      var ramUsed = hw.used_ram_mb || 0;
      document.getElementById('ram-used').textContent = formatMB(ramUsed);
      var ramPct = hw.total_ram_mb > 0 ? (ramUsed / hw.total_ram_mb * 100) : 0;
      document.getElementById('ram-bar').style.width = ramPct.toFixed(1) + '%';
      if (ramPct > 90) document.getElementById('ram-bar').className = 'fill red';
      else if (ramPct > 70) document.getElementById('ram-bar').className = 'fill orange';
      else document.getElementById('ram-bar').className = 'fill green';
    }
    if (hw.total_disk_mb) {
      document.getElementById('disk-total').textContent = '/ ' + formatMB(hw.total_disk_mb);
      var diskUsed = hw.used_disk_mb || 0;
      document.getElementById('disk-used').textContent = formatMB(diskUsed);
      var diskPct = hw.total_disk_mb > 0 ? (diskUsed / hw.total_disk_mb * 100) : 0;
      document.getElementById('disk-bar').style.width = diskPct.toFixed(1) + '%';
    }
  }

  if (data.hosted_shards !== undefined) {
    document.getElementById('hosted-shards').textContent = data.hosted_shards;
  }

  if (data.credits) {
    document.getElementById('credit-balance').textContent = data.credits.balance.toLocaleString();
    document.getElementById('stat-credits').textContent = data.credits.balance.toLocaleString();
    document.getElementById('credit-earned').textContent = '+' + (data.credits.lifetime_earned || 0).toLocaleString();
    document.getElementById('credit-spent').textContent = '-' + (data.credits.lifetime_spent || 0).toLocaleString();
  }
}

function updateStats(data) {
  if (data.peers !== undefined) {
    document.getElementById('stat-peers').textContent = data.peers;
  }
  if (data.credits !== undefined) {
    var bal = typeof data.credits === 'object' ? data.credits.balance : data.credits;
    document.getElementById('stat-credits').textContent = bal.toLocaleString();
    creditHistory.push(Math.abs(bal));
    if (creditHistory.length > 30) creditHistory.shift();
    renderSparkline('credit-sparkline', creditHistory);
  }
  if (data.requests_served !== undefined) {
    document.getElementById('stat-served').textContent = data.requests_served;
  }
  if (data.active_requests !== undefined) {
    document.getElementById('stat-active').textContent = data.active_requests;
  }
}

// --- Governance ---

async function loadGovernanceData() {
  // Governance role
  try {
    var resp = await fetch('/api/admin/governance/role');
    var role = await resp.json();
    var el = document.getElementById('governance-role');
    if (role.role) {
      el.textContent = capitalize(role.role);
    }
  } catch (e) {}

  // Proposals
  try {
    var resp = await fetch('/api/admin/proposals');
    var proposals = await resp.json();
    var list = document.getElementById('proposals-list');
    if (proposals && proposals.length > 0) {
      list.innerHTML = '';
      proposals.slice(0, 5).forEach(function (p) {
        var div = document.createElement('div');
        div.className = 'flex-between mb-1';
        div.innerHTML = '<span style="font-size:0.85rem">' + escapeHtml(p.title || p.proposal_type || 'Proposal') + '</span>' +
          '<span class="mono text-muted" style="font-size:0.8rem">' + (p.votes_for || 0) + '/' + (p.votes_against || 0) + '</span>';
        list.appendChild(div);
      });
    }
  } catch (e) {}

  // Issues
  try {
    var resp = await fetch('/api/admin/issues');
    var issues = await resp.json();
    var list = document.getElementById('issues-list');
    if (issues && issues.length > 0) {
      list.innerHTML = '';
      issues.slice(0, 5).forEach(function (issue) {
        var div = document.createElement('div');
        div.className = 'flex-between mb-1';
        div.innerHTML = '<span style="font-size:0.85rem">' + escapeHtml(issue.title || 'Issue') + '</span>' +
          '<span class="mono text-muted" style="font-size:0.8rem">' + (issue.upvotes || 0) + ' upvotes</span>';
        list.appendChild(div);
      });
    }
  } catch (e) {}

  // Governance params
  try {
    var resp = await fetch('/api/admin/governance/params');
    var params = await resp.json();
    var el = document.getElementById('gov-params');
    var lines = [];
    if (params.proposal_quorum !== undefined) lines.push('Quorum: ' + params.proposal_quorum);
    if (params.proposal_pass_threshold !== undefined) lines.push('Pass threshold: ' + (params.proposal_pass_threshold * 100).toFixed(0) + '%');
    if (params.release_approval_threshold !== undefined) lines.push('Release approvals: ' + params.release_approval_threshold);
    if (params.voting_duration_hours !== undefined) lines.push('Voting period: ' + params.voting_duration_hours + 'h');
    el.textContent = lines.length > 0 ? lines.join(' | ') : 'Default parameters';
  } catch (e) {}
}

// --- Network ---

async function loadNetworkData() {
  // Peers
  try {
    var resp = await fetch('/api/admin/peers');
    var peers = await resp.json();
    var list = document.getElementById('peers-list');
    if (peers && peers.length > 0) {
      list.innerHTML = '';
      peers.forEach(function (p) {
        var div = document.createElement('div');
        div.style.cssText = 'margin-bottom:10px;padding:8px 10px;background:var(--bg-tertiary);border-radius:var(--radius);border:1px solid var(--border)';

        var statusDot = '<span class="status-dot ' + (p.healthy ? 'online' : 'degraded') + '"></span>';
        var nodeId = '<span class="mono" style="font-size:0.8rem">' + escapeHtml(p.node_id || 'unknown') + '</span>';

        var details = '';
        if (p.gpu) {
          details += '<div style="font-size:0.75rem;color:var(--text-secondary);margin-top:3px">GPU: ' + escapeHtml(p.gpu) + '</div>';
        }
        if (p.hosted_models && p.hosted_models.length > 0) {
          details += '<div style="font-size:0.75rem;margin-top:2px">';
          p.hosted_models.forEach(function (model) {
            details += '<span class="source-badge local" style="margin-right:4px">' + escapeHtml(model) + '</span>';
          });
          details += '</div>';
        } else {
          details += '<div style="font-size:0.75rem;color:var(--text-muted);margin-top:3px">No models loaded</div>';
        }

        div.innerHTML = statusDot + nodeId + details;
        list.appendChild(div);
      });
    }
  } catch (e) {}

  // Latest release
  try {
    var resp = await fetch('/api/admin/releases/latest');
    if (resp.ok) {
      var release = await resp.json();
      var el = document.getElementById('latest-release');
      if (release && release.version) {
        el.textContent = 'v' + release.version + (release.approved ? ' (approved)' : ' (pending)');
      } else {
        el.textContent = 'No releases yet';
      }
    } else {
      document.getElementById('latest-release').textContent = 'No releases yet';
    }
  } catch (e) {
    document.getElementById('latest-release').textContent = 'No releases yet';
  }
}

// --- Models table ---

function renderModelsTable(models) {
  var table = document.getElementById('models-table');
  var empty = document.getElementById('models-empty');
  var tbody = document.getElementById('models-table-body');

  if (!models || models.length === 0) {
    table.style.display = 'none';
    empty.style.display = '';
    return;
  }

  table.style.display = '';
  empty.style.display = 'none';
  tbody.innerHTML = '';

  models.forEach(function (m) {
    var source = m.source || 'local';
    var shards = m.shards || [];
    var shardCount = m.shard_count || 0;
    var hostedShards = m.hosted_shards || 0;
    var safeId = (m.id || '').replace(/[^a-zA-Z0-9]/g, '_');

    // Main model row
    var tr = document.createElement('tr');

    // Source badge
    var sourceBadge = '<span class="source-badge ' + source + '">' + source + '</span>';
    if (shardCount > 1) {
      sourceBadge += ' <span class="text-muted" style="font-size:0.7rem">' +
        hostedShards + '/' + shardCount + ' shards</span>';
    }

    // Shard map: visual representation of which shards are local/available/missing
    var shardMap = '';
    if (shardCount > 1 && shards.length > 0) {
      shardMap = '<div class="shard-map" style="display:flex;gap:2px;margin-top:4px">';
      shards.forEach(function (s) {
        var color = s.local ? 'var(--green)' : (s.holders > 0 ? 'var(--accent)' : 'var(--border)');
        var title = 'Shard ' + s.index + ' (' + formatBytes(s.size_bytes) + ')' +
          (s.local ? ' - Local' : '') +
          (s.holders > 0 ? ' - ' + s.holders + ' holder(s)' : ' - Unavailable');
        shardMap += '<div title="' + title + '" style="width:' + Math.max(6, Math.floor(80 / shardCount)) +
          'px;height:14px;border-radius:2px;background:' + color + ';cursor:help"></div>';
      });
      shardMap += '</div>';
    }

    // Availability column
    var availability = '';
    if (m.local && m.status === 'loaded') {
      availability = '<span class="status-dot online"></span><span class="text-green" style="font-size:0.8rem">Loaded</span>';
    } else if (hostedShards > 0 && hostedShards === shardCount) {
      availability = '<span class="status-dot online"></span><span style="font-size:0.8rem">All shards local</span>';
    } else if (hostedShards > 0) {
      availability = '<span class="status-dot degraded"></span><span style="font-size:0.8rem">' +
        hostedShards + '/' + shardCount + ' shards</span>';
    } else if (m.peers_hosting > 0) {
      availability = '<span class="status-dot online"></span><span style="font-size:0.8rem">' +
        m.peers_hosting + ' peer' + (m.peers_hosting > 1 ? 's' : '') + '</span>';
    } else {
      availability = '<span class="text-muted" style="font-size:0.8rem">Discovered</span>';
    }
    if (m.peers_hosting > 0 && m.local) {
      availability += '<br><span class="text-muted" style="font-size:0.75rem">+ ' +
        m.peers_hosting + ' peer' + (m.peers_hosting > 1 ? 's' : '') + '</span>';
    }
    availability += shardMap;

    // Action column
    var action = '';
    if (m.status === 'loaded') {
      action = '<span class="text-green" style="font-size:0.8rem;font-weight:600">Active</span>';
      if (shardCount > 1) {
        action += '<br><span class="text-muted" style="font-size:0.7rem">Seeding ' + shardCount + ' shards</span>';
      }
    } else if (activeAcquisitions[m.id]) {
      action = '<span class="text-muted" style="font-size:0.8rem">&#8593; See progress above</span>';
    } else if (source === 'network' || m.status === 'available' || m.status === 'partial') {
      // Show download controls with shard count option
      if (shardCount > 1) {
        var missingShards = shardCount - hostedShards;
        action = '<div style="display:flex;align-items:center;gap:6px">' +
          '<select id="shard-count-' + safeId + '" class="shard-select" style="width:60px;padding:2px 4px;font-size:0.75rem;border-radius:var(--radius);border:1px solid var(--border);background:var(--bg-secondary);color:var(--text-primary)">';
        for (var i = 1; i <= missingShards; i++) {
          var selected = (i === missingShards) ? ' selected' : '';
          action += '<option value="' + i + '"' + selected + '>' + i + '</option>';
        }
        action += '</select>' +
          '<button class="btn btn-sm btn-primary" onclick="requestModel(\'' +
          escapeHtml(m.id) + '\')">Get Shards</button></div>';
      } else {
        action = '<button class="btn btn-sm btn-primary" onclick="requestModel(\'' +
          escapeHtml(m.id) + '\')">Download</button>';
      }
    } else if (source === 'local' && m.status !== 'loaded') {
      action = '<span class="text-muted" style="font-size:0.8rem">Stored</span>';
    }

    tr.innerHTML = '<td><strong>' + escapeHtml(m.id || m.name) + '</strong>' +
      (shardCount > 1 ? '<br><span class="text-muted" style="font-size:0.7rem">' + shardCount + ' shards @ ' + formatBytes((m.total_size_bytes || 0) / shardCount) + ' each</span>' : '') +
      '</td>' +
      '<td>' + sourceBadge + '</td>' +
      '<td>' + formatBytes(m.total_size_bytes || 0) + '</td>' +
      '<td>' + availability + '</td>' +
      '<td>' + action + '</td>';
    tbody.appendChild(tr);
  });
}

var activeAcquisitions = {};

async function requestModel(modelId) {
  try {
    var resp = await fetch('/api/admin/models/' + encodeURIComponent(modelId) + '/add', { method: 'POST' });
    var data = await resp.json();
    if (data.status === 'acquiring') {
      activeAcquisitions[modelId] = { started: Date.now() };
      renderAcquisitionPanel(modelId, null);
    } else {
      showBanner('warning', data.message || 'Model acquisition unavailable');
    }
  } catch (e) {
    showBanner('error', 'Failed to request model: ' + e.message);
  }
}

function updateAcquisitionProgress(acquisitions) {
  if (!acquisitions || acquisitions.length === 0) return;

  acquisitions.forEach(function (status) {
    var modelId = status.model_id;
    if (!modelId) return;

    // Track it
    if (!activeAcquisitions[modelId]) {
      if (status.state === 'complete') return; // Already done, skip
      activeAcquisitions[modelId] = { started: Date.now() };
    }

    renderAcquisitionPanel(modelId, status);

    // Clean up on completion
    if (status.state === 'complete') {
      setTimeout(function () {
        delete activeAcquisitions[modelId];
        loadInitialData();
      }, 3000);
    } else if (status.state && status.state.failed) {
      setTimeout(function () {
        delete activeAcquisitions[modelId];
      }, 10000);
    }
  });
}

function renderAcquisitionPanel(modelId, status) {
  var safeId = modelId.replace(/[^a-zA-Z0-9]/g, '_');
  var panelId = 'acq-panel-' + safeId;
  var panel = document.getElementById(panelId);

  if (!panel) {
    // Create the acquisition progress panel in the status banner area
    var banner = document.getElementById('status-banner');
    panel = document.createElement('div');
    panel.id = panelId;
    panel.className = 'acq-panel';
    panel.style.cssText = 'background:var(--bg-secondary);border:1px solid var(--border);border-radius:var(--radius);padding:12px 16px;margin-bottom:8px';
    banner.appendChild(panel);
  }

  if (!status) {
    panel.innerHTML = '<div style="display:flex;align-items:center;gap:8px">' +
      '<div class="spinner"></div>' +
      '<strong>' + escapeHtml(modelId) + '</strong>' +
      '<span class="text-muted" style="font-size:0.8rem">Starting acquisition...</span></div>';
    return;
  }

  var state = status.state;
  var stateName = typeof state === 'string' ? state : (state && state.failed ? 'failed' : 'unknown');
  var totalBytes = status.total_bytes || 0;
  var dlBytes = status.downloaded_bytes || 0;
  var pct = totalBytes > 0 ? Math.round((dlBytes / totalBytes) * 100) : 0;
  var speed = status.speed_bytes_per_sec || 0;
  var totalShards = status.total_shards || 0;
  var dlShards = status.downloaded_shards || 0;
  var verifiedShards = status.verified_shards || 0;
  var failedShards = status.failed_shards || 0;
  var shardProgress = status.shard_progress || {};
  var logs = status.log || [];

  // Header line
  var stateIcon = '&#9660;'; // downloading arrow
  var stateColor = 'var(--accent)';
  if (stateName === 'complete') { stateIcon = '&#10003;'; stateColor = 'var(--green)'; }
  else if (stateName === 'failed') { stateIcon = '&#10007;'; stateColor = 'var(--red)'; }
  else if (stateName === 'awaiting_manifest') { stateIcon = '&#8987;'; stateColor = 'var(--text-muted)'; }

  var header = '<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px">' +
    '<div><span style="color:' + stateColor + ';font-size:1.1rem;margin-right:6px">' + stateIcon + '</span>' +
    '<strong>' + escapeHtml(modelId) + '</strong>' +
    '<span class="text-muted" style="margin-left:8px;font-size:0.8rem">' + capitalize(stateName.replace('_', ' ')) + '</span></div>' +
    '<div class="mono" style="font-size:0.85rem">' +
    formatBytes(dlBytes) + ' / ' + formatBytes(totalBytes) + ' (' + pct + '%)' +
    (speed > 0 ? ' &mdash; ' + formatSpeed(speed) : '') +
    '</div></div>';

  // Main progress bar
  var barColor = stateName === 'complete' ? 'var(--green)' : (stateName === 'failed' ? 'var(--red)' : 'var(--accent)');
  var progressBar = '<div style="width:100%;height:6px;background:var(--bg-tertiary);border-radius:3px;overflow:hidden;margin-bottom:10px">' +
    '<div style="width:' + pct + '%;height:100%;background:' + barColor + ';transition:width 0.3s ease"></div></div>';

  // Per-shard progress grid
  var shardGrid = '';
  if (totalShards > 1) {
    shardGrid = '<div style="display:flex;flex-wrap:wrap;gap:4px;margin-bottom:10px">';
    for (var i = 0; i < totalShards; i++) {
      var sp = shardProgress[String(i)] || shardProgress[i];
      var sState = sp ? sp.state : 'pending';
      var sPct = (sp && sp.total_bytes > 0) ? Math.round((sp.downloaded_bytes / sp.total_bytes) * 100) : 0;

      var sColor = 'var(--bg-tertiary)';
      var sBorder = 'var(--border)';
      var sLabel = 'Pending';
      if (sState === 'complete') { sColor = 'var(--green)'; sBorder = 'var(--green)'; sLabel = 'Complete'; }
      else if (sState === 'downloading') { sColor = 'var(--accent)'; sBorder = 'var(--accent)'; sLabel = sPct + '%'; }
      else if (sState === 'verifying') { sColor = 'var(--yellow, #e6a817)'; sBorder = 'var(--yellow, #e6a817)'; sLabel = 'Verifying'; }
      else if (sState === 'failed') { sColor = 'var(--red)'; sBorder = 'var(--red)'; sLabel = 'Failed'; }

      var w = Math.max(28, Math.floor(100 / totalShards)) + 'px';
      var tooltip = 'Shard ' + i + ': ' + sLabel;
      if (sp) tooltip += ' (' + formatBytes(sp.downloaded_bytes || 0) + '/' + formatBytes(sp.total_bytes || 0) + ')';

      // Partially filled shard block for downloading state
      var innerFill = '';
      if (sState === 'downloading' && sPct > 0 && sPct < 100) {
        innerFill = '<div style="position:absolute;bottom:0;left:0;width:100%;height:' + sPct + '%;background:' + sColor + ';opacity:0.5;border-radius:0 0 3px 3px"></div>';
      }

      shardGrid += '<div title="' + tooltip + '" style="position:relative;width:' + w + ';height:24px;border-radius:3px;' +
        'border:1px solid ' + sBorder + ';overflow:hidden;cursor:help;text-align:center;line-height:24px;font-size:0.65rem;color:var(--text-secondary);' +
        (sState === 'complete' || sState === 'failed' || sState === 'verifying' ? 'background:' + sColor + ';color:#fff;font-weight:600' : 'background:var(--bg-tertiary)') +
        '">' + innerFill + '<span style="position:relative;z-index:1">' + i + '</span></div>';
    }
    shardGrid += '</div>';

    // Shard counter line
    shardGrid += '<div class="text-muted" style="font-size:0.75rem;margin-bottom:8px">' +
      'Shards: ' + verifiedShards + ' verified';
    if (failedShards > 0) shardGrid += ', <span style="color:var(--red)">' + failedShards + ' failed</span>';
    var inProgress = totalShards - verifiedShards - failedShards;
    if (inProgress > 0 && stateName !== 'complete') shardGrid += ', ' + inProgress + ' remaining';
    // ETA
    if (speed > 0 && dlBytes < totalBytes) {
      var remaining = totalBytes - dlBytes;
      var etaSec = Math.round(remaining / speed);
      shardGrid += ' &mdash; ETA: ' + formatEta(etaSec);
    }
    shardGrid += '</div>';
  }

  // Log tail (last 6 lines)
  var logHtml = '';
  if (logs.length > 0) {
    var recentLogs = logs.slice(-6);
    logHtml = '<div style="font-family:var(--mono);font-size:0.72rem;color:var(--text-secondary);background:var(--bg-tertiary);padding:6px 10px;border-radius:var(--radius);max-height:120px;overflow-y:auto">';
    recentLogs.forEach(function (line) {
      logHtml += '<div style="margin-bottom:2px">' + escapeHtml(line) + '</div>';
    });
    logHtml += '</div>';
  }

  panel.innerHTML = header + progressBar + shardGrid + logHtml;
}

function formatSpeed(bytesPerSec) {
  if (bytesPerSec >= 1048576) return (bytesPerSec / 1048576).toFixed(1) + ' MB/s';
  if (bytesPerSec >= 1024) return Math.round(bytesPerSec / 1024) + ' KB/s';
  return bytesPerSec + ' B/s';
}

function formatEta(seconds) {
  if (seconds < 60) return seconds + 's';
  if (seconds < 3600) return Math.floor(seconds / 60) + 'm ' + (seconds % 60) + 's';
  var h = Math.floor(seconds / 3600);
  var m = Math.floor((seconds % 3600) / 60);
  return h + 'h ' + m + 'm';
}

// --- Settings ---

async function saveSettings() {
  var btn = event.target;
  btn.disabled = true;
  btn.textContent = 'Saving...';

  var config = {
    contribution: document.getElementById('settings-contribution').value,
    max_concurrent_requests: parseInt(document.getElementById('settings-max-requests').value, 10),
    max_bandwidth_mbps: parseInt(document.getElementById('settings-bandwidth').value, 10),
    max_disk_mb: parseInt(document.getElementById('settings-disk').value, 10),
  };

  try {
    var resp = await fetch('/api/admin/config', {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(config),
    });
    if (resp.ok) {
      showBanner('success', 'Settings saved successfully');
      setTimeout(function() { document.getElementById('status-banner').innerHTML = ''; }, 3000);
    } else {
      showBanner('error', 'Failed to save settings');
    }
  } catch (e) {
    showBanner('error', 'Error: ' + e.message);
  }

  btn.disabled = false;
  btn.textContent = 'Save Settings';
}

// --- Helpers ---

function escapeHtml(str) {
  var div = document.createElement('div');
  div.appendChild(document.createTextNode(str));
  return div.innerHTML;
}

function setTierBadge(elementId, tier) {
  var el = document.getElementById(elementId);
  el.textContent = capitalize(tier);
  el.className = 'tier-badge ' + tier.toLowerCase();
}

function formatUptime(seconds) {
  if (seconds < 60) return seconds + 's';
  if (seconds < 3600) return Math.floor(seconds / 60) + 'm';
  var h = Math.floor(seconds / 3600);
  var m = Math.floor((seconds % 3600) / 60);
  if (h >= 24) {
    var d = Math.floor(h / 24);
    h = h % 24;
    return d + 'd ' + h + 'h';
  }
  return h + 'h ' + m + 'm';
}

function formatMB(mb) {
  if (!mb || mb === 0) return '\u2014';
  if (mb >= 1024) return (mb / 1024).toFixed(1) + ' GB';
  return mb + ' MB';
}

function formatBytes(bytes) {
  if (!bytes || bytes === 0) return '\u2014';
  if (bytes >= 1073741824) return (bytes / 1073741824).toFixed(1) + ' GB';
  if (bytes >= 1048576) return (bytes / 1048576).toFixed(1) + ' MB';
  return Math.round(bytes / 1024) + ' KB';
}

function capitalize(s) {
  return s.charAt(0).toUpperCase() + s.slice(1);
}

function renderSparkline(containerId, data) {
  var container = document.getElementById(containerId);
  if (!data || data.length === 0) return;
  var max = Math.max.apply(null, data) || 1;
  container.innerHTML = '';
  data.forEach(function (val) {
    var bar = document.createElement('div');
    bar.className = 'bar';
    bar.style.height = Math.max(2, (val / max) * 36) + 'px';
    container.appendChild(bar);
  });
}

// --- Init ---
loadInitialData();
connectWebSocket();

// Refresh stats every 30 seconds
setInterval(loadInitialData, 30000);
