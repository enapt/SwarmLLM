'use strict';

let ws = null;
let creditHistory = [];

// --- Initial data load ---

async function loadInitialData() {
  try {
    var resp = await fetch('/api/admin/stats');
    var data = await resp.json();
    updateDashboard(data);
  } catch (e) {
    console.error('Failed to load initial stats:', e);
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
}

// --- WebSocket for real-time updates ---

function connectWebSocket() {
  var protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  ws = new WebSocket(protocol + '//' + window.location.host + '/api/admin/ws');

  ws.onmessage = function (event) {
    try {
      var msg = JSON.parse(event.data);
      handleWsMessage(msg);
    } catch (e) {
      console.warn('Invalid WS message:', event.data);
    }
  };

  ws.onclose = function () {
    // Reconnect after 3 seconds
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
    case 'inference_progress':
      updateActiveRequest(msg.data);
      break;
    case 'peer_joined':
      // Could show a toast notification
      break;
    case 'shard_status':
      break;
  }
}

// --- Dashboard update ---

function updateDashboard(data) {
  // Node info
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

  // Stats
  updateStats(data);

  // Hardware
  if (data.hardware) {
    var hw = data.hardware;
    document.getElementById('node-gpu').textContent = hw.gpu_name || 'CPU only';
    if (hw.total_ram_mb) {
      document.getElementById('ram-total').textContent = '/ ' + formatMB(hw.total_ram_mb);
      var ramUsed = hw.used_ram_mb || 0;
      document.getElementById('ram-used').textContent = formatMB(ramUsed);
      var ramPct = hw.total_ram_mb > 0 ? (ramUsed / hw.total_ram_mb * 100) : 0;
      document.getElementById('ram-bar').style.width = ramPct + '%';
    }
    if (hw.total_disk_mb) {
      document.getElementById('disk-total').textContent = '/ ' + formatMB(hw.total_disk_mb);
      var diskUsed = hw.used_disk_mb || 0;
      document.getElementById('disk-used').textContent = formatMB(diskUsed);
      var diskPct = hw.total_disk_mb > 0 ? (diskUsed / hw.total_disk_mb * 100) : 0;
      document.getElementById('disk-bar').style.width = diskPct + '%';
    }
  }

  if (data.hosted_shards !== undefined) {
    document.getElementById('hosted-shards').textContent = data.hosted_shards;
  }

  // Credits
  if (data.credits) {
    document.getElementById('credit-balance').textContent = data.credits.balance;
    document.getElementById('stat-credits').textContent = data.credits.balance;
    document.getElementById('credit-earned').textContent = '+' + (data.credits.lifetime_earned || 0);
    document.getElementById('credit-spent').textContent = '-' + (data.credits.lifetime_spent || 0);
  }
}

function updateStats(data) {
  if (data.peers !== undefined) {
    document.getElementById('stat-peers').textContent = data.peers;
  }
  if (data.credits !== undefined) {
    var bal = typeof data.credits === 'object' ? data.credits.balance : data.credits;
    document.getElementById('stat-credits').textContent = bal;
    // Push to sparkline history
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

function updateActiveRequest(data) {
  var container = document.getElementById('active-requests');
  var id = 'req-' + data.request_id;
  var el = document.getElementById(id);
  if (!el) {
    el = document.createElement('div');
    el.id = id;
    el.className = 'flex-between mb-1';
    container.innerHTML = '';
    container.appendChild(el);
  }
  var pct = data.tokens_total > 0 ? Math.round(data.tokens_generated / data.tokens_total * 100) : 0;
  el.innerHTML = '<span class="mono">' + data.request_id.substring(0, 8) + '...</span>' +
    '<div style="flex:1;margin:0 12px"><div class="progress-bar"><div class="fill accent" style="width:' + pct + '%"></div></div></div>' +
    '<span class="mono">' + data.tokens_generated + '/' + data.tokens_total + '</span>';
}

// --- Models table ---

function renderModelsTable(models) {
  var tbody = document.getElementById('models-table-body');
  if (!models || models.length === 0) {
    tbody.innerHTML = '<tr><td colspan="5" class="text-muted">No models loaded</td></tr>';
    return;
  }
  tbody.innerHTML = '';
  models.forEach(function (m) {
    var tr = document.createElement('tr');
    tr.innerHTML = '<td>' + m.id + '</td>' +
      '<td>' + formatBytes(m.total_size_bytes || 0) + '</td>' +
      '<td>' + (m.shard_count || '—') + '</td>' +
      '<td><span class="status-dot ' + (m.healthy ? 'online' : 'degraded') + '"></span>' + (m.healthy ? 'Healthy' : 'Degraded') + '</td>' +
      '<td>' + (m.status || 'available') + '</td>';
    tbody.appendChild(tr);
  });
}

// --- Settings ---

async function saveSettings() {
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
      alert('Settings saved.');
    } else {
      alert('Failed to save settings.');
    }
  } catch (e) {
    alert('Error saving settings: ' + e.message);
  }
}

// --- Helpers ---

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
  return h + 'h ' + m + 'm';
}

function formatMB(mb) {
  if (!mb || mb === 0) return '—';
  if (mb >= 1024) return (mb / 1024).toFixed(1) + ' GB';
  return mb + ' MB';
}

function formatBytes(bytes) {
  if (!bytes || bytes === 0) return '—';
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
    bar.style.height = Math.max(2, (val / max) * 30) + 'px';
    container.appendChild(bar);
  });
}

// --- Init ---
loadInitialData();
connectWebSocket();
