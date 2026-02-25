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
        div.className = 'flex-between mb-1';
        div.innerHTML = '<span class="mono" style="font-size:0.8rem">' + escapeHtml(p.node_id || p.peer_id || 'unknown') + '</span>' +
          '<span class="status-dot ' + (p.healthy ? 'online' : 'degraded') + '"></span>';
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
    var tr = document.createElement('tr');
    var statusClass = m.healthy ? 'online' : 'degraded';
    var statusText = m.healthy ? 'Healthy' : 'Degraded';
    tr.innerHTML = '<td>' + escapeHtml(m.id) + '</td>' +
      '<td>' + formatBytes(m.total_size_bytes || 0) + '</td>' +
      '<td>' + (m.shard_count || '\u2014') + '</td>' +
      '<td><span class="status-dot ' + statusClass + '"></span>' + statusText + '</td>' +
      '<td>' + capitalize(m.status || 'available') + '</td>';
    tbody.appendChild(tr);
  });
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
